import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html, no_update


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "sp500_clean.csv"
STARTING_CAPITAL = 10_000.0


COLORS = {
    "bg": "#f6f4ef",
    "panel": "#ffffff",
    "ink": "#1f2933",
    "muted": "#68737d",
    "line": "#d8d5cd",
    "accent": "#167c80",
    "accent_dark": "#0f5d61",
    "buy": "#188a4d",
    "sell": "#b73e3e",
    "gold": "#c28a1d",
    "blue": "#2f5f98",
}


def load_market_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["daily_return"] = df["close"].pct_change().fillna(0.0)
    df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
    df["volatility_20"] = df["log_return"].rolling(20, min_periods=5).std().fillna(0.0)
    return df


MARKET_DF = load_market_data()


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def metric_card(label, value, tone=None):
    border = COLORS["line"]
    if tone == "good":
        border = COLORS["buy"]
    elif tone == "bad":
        border = COLORS["sell"]
    return html.Div(
        [html.Div(label, className="metric-label"), html.Div(value, className="metric-value")],
        className="metric-card",
        style={"borderTopColor": border},
    )


def simulate_strategy(short_window: int, long_window: int, stop_loss: float, take_profit: float, end_idx: int):
    df = MARKET_DF.iloc[: end_idx + 1].copy()
    df["ma_short"] = df["close"].rolling(short_window, min_periods=1).mean()
    df["ma_long"] = df["close"].rolling(long_window, min_periods=1).mean()
    df["signal"] = (df["ma_short"] > df["ma_long"]).astype(int)

    cash = STARTING_CAPITAL
    shares = 0.0
    entry_price = None
    entry_date = None
    trade_rows = []
    equity_rows = []

    for i, row in df.iterrows():
        price = float(row["close"])
        date = row["date"]
        desired_long = int(row["signal"]) == 1 and i >= max(short_window, long_window)

        exit_reason = None
        if shares and entry_price:
            trade_return = (price - entry_price) / entry_price
            if trade_return <= -stop_loss:
                exit_reason = "Stop loss"
            elif trade_return >= take_profit:
                exit_reason = "Take profit"
            elif not desired_long:
                exit_reason = "MA crossover"

        if shares and exit_reason:
            cash = shares * price
            pnl = cash - STARTING_CAPITAL if not trade_rows else cash - trade_rows[-1]["balance_after_entry"]
            trade_rows.append(
                {
                    "date": date,
                    "side": "SELL",
                    "price": price,
                    "shares": shares,
                    "reason": exit_reason,
                    "pnl": pnl,
                    "equity": cash,
                    "entry_date": entry_date,
                    "balance_after_entry": cash,
                }
            )
            shares = 0.0
            entry_price = None
            entry_date = None

        if desired_long and not shares:
            shares = cash / price
            entry_price = price
            entry_date = date
            trade_rows.append(
                {
                    "date": date,
                    "side": "BUY",
                    "price": price,
                    "shares": shares,
                    "reason": "Short MA crossed above long MA",
                    "pnl": 0.0,
                    "equity": cash,
                    "entry_date": date,
                    "balance_after_entry": cash,
                }
            )
            cash = 0.0

        equity = cash + shares * price
        equity_rows.append({"date": date, "equity": equity, "close": price, "in_market": bool(shares)})

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)

    if equity_df.empty:
        equity_df = pd.DataFrame({"date": df["date"], "equity": STARTING_CAPITAL, "close": df["close"], "in_market": False})

    return df, equity_df, trades_df


def summarize(equity_df: pd.DataFrame, trades_df: pd.DataFrame):
    final_equity = float(equity_df["equity"].iloc[-1])
    total_return = final_equity / STARTING_CAPITAL - 1
    equity_returns = equity_df["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    downside = equity_returns[equity_returns < 0]
    volatility = float(equity_returns.std())
    sharpe = 0.0 if volatility == 0 else float((equity_returns.mean() / volatility) * math.sqrt(252))
    running_max = equity_df["equity"].cummax()
    max_drawdown = float(((equity_df["equity"] / running_max) - 1).min())
    sells = trades_df[trades_df["side"] == "SELL"] if not trades_df.empty else pd.DataFrame()
    win_rate = 0.0 if sells.empty else float((sells["pnl"] > 0).mean())
    invested_days = int(equity_df["in_market"].sum())
    buy_hold = float(equity_df["close"].iloc[-1] / equity_df["close"].iloc[0] - 1)
    sortino = 0.0 if downside.std() == 0 or np.isnan(downside.std()) else float((equity_returns.mean() / downside.std()) * math.sqrt(252))

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "buy_hold": buy_hold,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trades": int(len(sells)),
        "invested_days": invested_days,
    }


def build_chart(df: pd.DataFrame, equity_df: pd.DataFrame, trades_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="S&P 500",
            increasing_line_color=COLORS["buy"],
            decreasing_line_color=COLORS["sell"],
        )
    )
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma_short"], mode="lines", name="Short MA", line={"color": COLORS["blue"], "width": 1.8}))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma_long"], mode="lines", name="Long MA", line={"color": COLORS["gold"], "width": 1.8}))

    if not trades_df.empty:
        buys = trades_df[trades_df["side"] == "BUY"]
        sells = trades_df[trades_df["side"] == "SELL"]
        fig.add_trace(
            go.Scatter(
                x=buys["date"],
                y=buys["price"],
                mode="markers",
                name="Buy",
                marker={"symbol": "triangle-up", "size": 12, "color": COLORS["buy"], "line": {"width": 1, "color": "#ffffff"}},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sells["date"],
                y=sells["price"],
                mode="markers",
                name="Sell",
                marker={"symbol": "triangle-down", "size": 12, "color": COLORS["sell"], "line": {"width": 1, "color": "#ffffff"}},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=equity_df["date"],
            y=equity_df["equity"],
            mode="lines",
            name="Strategy equity",
            yaxis="y2",
            line={"color": COLORS["accent"], "width": 2},
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=620,
        margin={"l": 20, "r": 30, "t": 20, "b": 20},
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": COLORS["ink"]},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        xaxis={"rangeslider": {"visible": False}, "showgrid": False},
        yaxis={"title": "Price", "gridcolor": "#ece8df"},
        yaxis2={"title": "Equity", "overlaying": "y", "side": "right", "showgrid": False},
        hovermode="x unified",
    )
    return fig


def build_trade_table(trades_df: pd.DataFrame):
    if trades_df.empty:
        return []
    out = trades_df.tail(12).copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["price"] = out["price"].map(lambda v: f"{v:,.2f}")
    out["shares"] = out["shares"].map(lambda v: f"{v:,.3f}")
    out["pnl"] = out["pnl"].map(lambda v: f"{v:,.2f}")
    out["equity"] = out["equity"].map(lambda v: f"{v:,.2f}")
    return out[["date", "side", "price", "shares", "reason", "pnl", "equity"]].to_dict("records")


dash_app = Dash(__name__, title="Trading Strategy Simulator")
server = dash_app.server
app = server


dash_app.layout = html.Div(
    [
        dcc.Interval(id="playback", interval=650, n_intervals=0, disabled=True),
        html.Div(
            [
                html.Div(
                    [
                        html.H1("Trading Strategy Simulator"),
                        html.P("A user-friendly dashboard for strategy design, execution logic, and performance review."),
                    ],
                    className="brand",
                ),
                html.Div([html.Span(className="status-dot"), html.Span("S&P 500 OHLCV data | moving-average signal engine | risk controls")], className="nav-note"),
            ],
            className="topbar",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2("Simulation Controls", className="section-title"),
                                html.Div(
                                    [
                                        html.Label("Playback date"),
                                        dcc.Slider(
                                            id="date-slider",
                                            min=80,
                                            max=len(MARKET_DF) - 1,
                                            value=80,
                                            step=5,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                    ],
                                    className="field",
                                ),
                                html.Div(
                                    [
                                        html.Label("Short moving average"),
                                        dcc.Slider(id="short-window", min=5, max=60, step=1, value=20, marks={5: "5", 20: "20", 40: "40", 60: "60"}),
                                    ],
                                    className="field",
                                ),
                                html.Div(
                                    [
                                        html.Label("Long moving average"),
                                        dcc.Slider(id="long-window", min=30, max=180, step=5, value=90, marks={30: "30", 90: "90", 180: "180"}),
                                    ],
                                    className="field",
                                ),
                                html.Div(
                                    [
                                        html.Label("Stop loss"),
                                        dcc.Slider(id="stop-loss", min=0.02, max=0.20, step=0.01, value=0.08, marks={0.02: "2%", 0.08: "8%", 0.20: "20%"}),
                                    ],
                                    className="field",
                                ),
                                html.Div(
                                    [
                                        html.Label("Take profit"),
                                        dcc.Slider(id="take-profit", min=0.04, max=0.40, step=0.01, value=0.16, marks={0.04: "4%", 0.16: "16%", 0.40: "40%"}),
                                    ],
                                    className="field",
                                ),
                                html.Div(
                                    [
                                        html.Button("Play", id="play-button", n_clicks=0),
                                        html.Button("Reset", id="reset-button", n_clicks=0, className="secondary"),
                                    ],
                                    className="button-row",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.H2("What You See", className="section-title"),
                                html.Ul(
                                    [
                                        html.Li("Data cleaning and time-ordered simulation discipline."),
                                        html.Li("Strategy logic with transparent entry, exit, and risk rules."),
                                        html.Li("Business metrics: return, drawdown, win rate, and trade history."),
                                        html.Li("Interactive product thinking rather than a static notebook screenshot."),
                                    ],
                                    className="story-list",
                                ),
                            ]
                        ),
                    ],
                    className="panel controls",
                ),
                html.Div(
                    [
                        html.Div(id="metrics", className="metrics"),
                        html.Div([dcc.Graph(id="trading-chart", config={"displayModeBar": True, "responsive": True})], className="chart-wrap"),
                    ],
                    className="panel",
                ),
            ],
            className="layout",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("Recent Trade Log", className="section-title"),
                        dash_table.DataTable(
                            id="trade-table",
                            columns=[
                                {"name": "Date", "id": "date"},
                                {"name": "Side", "id": "side"},
                                {"name": "Price", "id": "price"},
                                {"name": "Shares", "id": "shares"},
                                {"name": "Reason", "id": "reason"},
                                {"name": "P&L", "id": "pnl"},
                                {"name": "Equity", "id": "equity"},
                            ],
                            data=[],
                            page_size=12,
                            style_as_list_view=True,
                            style_table={"overflowX": "auto"},
                            style_cell={"padding": "10px", "fontSize": "13px", "textAlign": "left", "border": "0", "minWidth": "92px"},
                            style_header={"backgroundColor": "#fffaf0", "fontWeight": "800", "borderBottom": "1px solid #d8d5cd"},
                            style_data_conditional=[
                                {"if": {"filter_query": "{side} = BUY", "column_id": "side"}, "color": COLORS["buy"], "fontWeight": "800"},
                                {"if": {"filter_query": "{side} = SELL", "column_id": "side"}, "color": COLORS["sell"], "fontWeight": "800"},
                            ],
                        ),
                    ],
                    className="panel table-panel",
                ),
                html.Div(
                    [
                        html.H2("Project Story", className="section-title"),
                        html.P(
                            "This dashboard turns historical market data into an inspectable trading simulation. "
                            "The core signal is intentionally explainable: a short moving average crossing above a long moving average creates entry pressure, while stop-loss, take-profit, and crossover exits manage risk."
                        ),
                        html.P(
                            "For the wider XAI portfolio, this pairs well with the existing fairness-aware model work: the same discipline appears here as transparent assumptions, time-aware evaluation, readable metrics, and decisions a reviewer can audit."
                        ),
                        html.Ul(
                            [
                                html.Li("Built with Python, Dash, Plotly, pandas, and vectorized market features."),
                                html.Li("Designed as a portfolio artifact: interactive, measurable, and easy to discuss in interviews."),
                                html.Li("Extensible next steps: add ML probabilities, SHAP explanations, transaction costs, and walk-forward validation."),
                            ],
                            className="story-list",
                        ),
                    ],
                    className="panel story-panel",
                ),
            ],
            className="lower",
        ),
    ],
    className="page",
)


@dash_app.callback(
    Output("playback", "disabled"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    State("playback", "disabled"),
    prevent_initial_call=True,
)
def toggle_playback(_, disabled):
    is_disabled = not disabled
    return is_disabled, "Play" if is_disabled else "Pause"


@dash_app.callback(
    Output("date-slider", "value"),
    Input("playback", "n_intervals"),
    Input("reset-button", "n_clicks"),
    State("date-slider", "value"),
    prevent_initial_call=True,
)
def advance_playback(_, reset_clicks, current):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger == "reset-button":
        return 80
    return min(current + 8, len(MARKET_DF) - 1)


@dash_app.callback(
    Output("trading-chart", "figure"),
    Output("metrics", "children"),
    Output("trade-table", "data"),
    Input("date-slider", "value"),
    Input("short-window", "value"),
    Input("long-window", "value"),
    Input("stop-loss", "value"),
    Input("take-profit", "value"),
)
def update_dashboard(end_idx, short_window, long_window, stop_loss, take_profit):
    if short_window >= long_window:
        long_window = short_window + 10

    df, equity_df, trades_df = simulate_strategy(short_window, long_window, stop_loss, take_profit, end_idx)
    stats = summarize(equity_df, trades_df)
    current_date = df["date"].iloc[-1].strftime("%Y-%m-%d")

    metrics = [
        metric_card("Portfolio Value", money(stats["final_equity"]), "good" if stats["total_return"] >= 0 else "bad"),
        metric_card("Strategy Return", pct(stats["total_return"]), "good" if stats["total_return"] >= stats["buy_hold"] else "bad"),
        metric_card("Buy & Hold", pct(stats["buy_hold"])),
        metric_card("Max Drawdown", pct(stats["max_drawdown"]), "bad" if stats["max_drawdown"] < -0.15 else None),
        metric_card("Sharpe", f"{stats['sharpe']:.2f}"),
        metric_card("Sortino", f"{stats['sortino']:.2f}"),
        metric_card("Win Rate", pct(stats["win_rate"])),
        metric_card("As Of", current_date),
    ]

    return build_chart(df, equity_df, trades_df), metrics, build_trade_table(trades_df)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    dash_app.run(host="0.0.0.0", port=port, debug=False)
