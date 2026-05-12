import argparse
import logging
import sys

from database import setup_database


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def cmd_dashboard(args) -> None:
    from web_dashboard import run
    setup_database()
    run()


def cmd_collect(args) -> None:
    from data_collector import run_collection_loop
    setup_database()
    run_collection_loop()


def cmd_paper(args) -> None:
    from paper_trader import run_iteration
    setup_database()
    run_iteration()


def cmd_history(args) -> None:
    from data_collector import run_historical_download
    setup_database()
    tf = args.timeframe
    days = args.days
    print(f"\nLade historische Daten: {days} Tage zurück, Zeitrahmen {tf}")
    print("Abbruch mit Ctrl+C möglich.\n")
    total = run_historical_download(timeframe=tf, days_back=days)
    print(f"\nFertig. {total} Candles in der Datenbank gespeichert.")


def cmd_trade(args) -> None:
    import time
    from paper_trader import run_iteration
    from config import COLLECT_INTERVAL_SECONDS
    setup_database()
    logger = logging.getLogger("trade")
    logger.info("Paper Trading Loop gestartet (Intervall: %ds). Ctrl+C zum Stoppen.", COLLECT_INTERVAL_SECONDS)
    while True:
        try:
            run_iteration()
        except Exception as e:
            logger.error("Trading-Iteration fehlgeschlagen: %s", e)
        time.sleep(COLLECT_INTERVAL_SECONDS)


def cmd_backtest(args) -> None:
    from backtester import run_backtest
    setup_database()

    timeframe = getattr(args, "timeframe", "1h")
    metrics = run_backtest(timeframe=timeframe)

    if "error" in metrics:
        print(f"\nBacktest failed: {metrics['error']}")
        print("Tip: run 'python main.py collect' first to gather historical data.")
        return

    print("\n=== Backtest Results ===")
    print(f"  Timeframe       : {timeframe}")
    print(f"  Start Capital   : ${metrics['start_capital']:.2f}")
    print(f"  End Capital     : ${metrics['end_capital']:.2f}")
    print(f"  Total Return    : {metrics['total_return_pct']:+.2f}%")
    print(f"  Trade Count     : {metrics['trade_count']}")
    print(f"  Win Count       : {metrics['win_count']}")
    print(f"  Win Rate        : {metrics['win_rate']:.1f}%")
    print(f"  Max Drawdown    : {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Best Trade      : {metrics['best_trade_pct']:+.2f}%")
    print(f"  Worst Trade     : {metrics['worst_trade_pct']:+.2f}%")
    print()


def cmd_train(args) -> None:
    setup_database()
    timeframe = getattr(args, "timeframe", "1h")

    if getattr(args, "validate", False):
        from model_trainer import validate_walk_forward
        from config import WALK_FORWARD_FOLDS
        print(f"\nWalk-Forward Validation: {WALK_FORWARD_FOLDS} folds, timeframe={timeframe}\n")
        try:
            validate_walk_forward(timeframe=timeframe)
        except ValueError as e:
            print(f"\nValidation failed: {e}")
        return

    from model_trainer import train
    print(f"\nTraining XGBoost model on {timeframe} candles...\n")
    try:
        train(timeframe=timeframe)
    except ValueError as e:
        print(f"\nTraining failed: {e}")


def cmd_tune(args) -> None:
    from model_trainer import tune
    from config import OPTUNA_TRIALS
    setup_database()
    timeframe = getattr(args, "timeframe", "1h")
    n_trials = getattr(args, "trials", None) or OPTUNA_TRIALS
    try:
        tune(timeframe=timeframe, n_trials=n_trials)
    except ValueError as e:
        print(f"\nTuning failed: {e}")


def cmd_status(args) -> None:
    from paper_trader import get_portfolio_status
    setup_database()

    status = get_portfolio_status()
    print("\n=== Portfolio Status ===")
    print(f"  Cash            : ${status['cash']:.4f}")
    print(f"  IOTA Holdings   : {status['iota_holdings']:.6f} IOTA")
    print(f"  Entry Price     : {'${:.4f}'.format(status['entry_price']) if status['entry_price'] else 'None'}")
    print(f"  Current Price   : ${status['current_price']:.4f}")
    print(f"  Portfolio Value : ${status['portfolio_value']:.4f}")
    print(f"  Unrealized PnL  : ${status['unrealized_pnl']:+.4f}")
    print(f"  Realized PnL    : ${status['realized_pnl']:+.4f}")
    print(f"  Total PnL       : ${status['total_pnl']:+.4f}")
    print(f"  Trade Count     : {status['trade_count']}")
    print()


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="IOTA Paper Trading Bot — Bitfinex + Keras LSTM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  collect    Start continuous data collection (runs until stopped with Ctrl+C)
  train      Train the Keras LSTM model on stored historical candles
  paper      Execute one paper trading iteration using current market data
  backtest   Run a historical backtest on stored candle data
  status     Show current portfolio summary

Typical workflow:
  1. python main.py collect          # collect data for a few hours/days
  2. python main.py train            # train the LSTM model
  3. python main.py paper            # run paper trading (loops or one-shot)
  4. python main.py backtest         # evaluate strategy on history
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("collect", help="Start live data collection loop")

    tr_parser = subparsers.add_parser("train", help="Train XGBoost model on stored candles")
    tr_parser.add_argument(
        "--timeframe", default="1h", choices=["1m", "5m", "1h"],
        help="Candle timeframe to train on (default: 1h)",
    )
    tr_parser.add_argument(
        "--validate", action="store_true",
        help="Run walk-forward validation instead of training (no model saved)",
    )

    tune_parser = subparsers.add_parser("tune", help="Tune XGBoost hyperparameters with Optuna")
    tune_parser.add_argument(
        "--timeframe", default="1h", choices=["1m", "5m", "1h"],
        help="Candle timeframe (default: 1h)",
    )
    tune_parser.add_argument(
        "--trials", type=int, default=None,
        help="Number of Optuna trials (default from config)",
    )

    subparsers.add_parser("paper", help="Run one paper trading iteration")
    subparsers.add_parser("trade", help="Start continuous paper trading loop")

    hist_parser = subparsers.add_parser("history", help="Bulk-download historical candles from Bitfinex")
    hist_parser.add_argument("--timeframe", default="1h", choices=["1m", "5m", "1h"],
                             help="Candle timeframe (default: 1h)")
    hist_parser.add_argument("--days", type=int, default=365,
                             help="How many days back to download (default: 365)")

    bt_parser = subparsers.add_parser("backtest", help="Run historical backtest")
    bt_parser.add_argument(
        "--timeframe", default="1h", choices=["1m", "5m", "1h"],
        help="Candle timeframe to use (default: 1h)",
    )

    subparsers.add_parser("status", help="Show portfolio status")

    subparsers.add_parser("dashboard", help="Start web dashboard (http://localhost:5000)")

    dispatch = {
        "collect": cmd_collect,
        "history": cmd_history,
        "train": cmd_train,
        "tune": cmd_tune,
        "paper": cmd_paper,
        "trade": cmd_trade,
        "backtest": cmd_backtest,
        "status": cmd_status,
        "dashboard": cmd_dashboard,
    }

    args = parser.parse_args()
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
