import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, default="baseline",
                   help="Run folder name under runs/<algo>/ (e.g., baseline)")
    p.add_argument("--runs_dir", type=str, default="runs",
                   help="Base runs directory (default: runs)")
    p.add_argument("--show", action="store_true",
                   help="Show figures on screen (still saves files).")
    p.add_argument("--dpi", type=int, default=150,
                   help="Saved figure DPI.")
    return p.parse_args()


def _safe_read_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def _pct(x: pd.Series, q: float) -> float:
    return float(x.quantile(q))


def summarize_training(df, label: str) -> str:
    if df is None or df.empty:
        return f"{label} TRAIN | (missing metrics.csv)"
    needed = {"episode", "score", "mean100"}
    missing = needed - set(df.columns)
    if missing:
        return f"{label} TRAIN | metrics.csv missing columns: {sorted(missing)}"

    last_ep = int(df["episode"].iloc[-1])
    last_mean100 = float(df["mean100"].iloc[-1])
    best_score = int(df["score"].max())
    avg_last_100 = float(df["score"].tail(100).mean()) if len(df) >= 1 else float("nan")

    return (
        f"{label} TRAIN | episodes={last_ep} "
        f"best_score={best_score} "
        f"mean100_last={last_mean100:.2f} "
        f"avg_score_last100={avg_last_100:.2f}"
    )


def summarize_eval(df: pd.DataFrame, label: str) -> str:
    if df is None or df.empty:
        return f"{label} EVAL  | (missing eval.csv)"
    needed = {"score", "frames_alive"}
    missing = needed - set(df.columns)
    if missing:
        return f"{label} EVAL  | eval.csv missing columns: {sorted(missing)}"

    score = df["score"].astype(float)
    frames = df["frames_alive"].astype(float)

    return (
        f"{label} EVAL  | n={len(df)} "
        f"score_mean={score.mean():.2f} score_std={score.std(ddof=1):.2f} "
        f"score_med={score.median():.2f} score_p25={_pct(score, 0.25):.2f} score_p75={_pct(score, 0.75):.2f} "
        f"frames_mean={frames.mean():.1f} frames_std={frames.std(ddof=1):.1f} "
        f"frames_med={frames.median():.1f}"
    )


def _save_or_show(fig_name: str, out_dir: Path, show: bool, dpi: int):
    out_path = out_dir / fig_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    return out_path


def plot_training(dqn_df: pd.DataFrame, ddqn_df: pd.DataFrame,
                  out_dir: Path, show: bool, dpi: int):
    # 1) Score over episodes
    plt.figure()
    if dqn_df is not None and not dqn_df.empty and {"episode", "score"} <= set(dqn_df.columns):
        plt.plot(dqn_df["episode"], dqn_df["score"], label="DQN")
    if ddqn_df is not None and not ddqn_df.empty and {"episode", "score"} <= set(ddqn_df.columns):
        plt.plot(ddqn_df["episode"], ddqn_df["score"], label="Double DQN")
    plt.title("Training: Score per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    p1 = _save_or_show("train_score.png", out_dir, show, dpi)

    # 2) Mean100 over episodes
    plt.figure()
    if dqn_df is not None and not dqn_df.empty and {"episode", "mean100"} <= set(dqn_df.columns):
        plt.plot(dqn_df["episode"], dqn_df["mean100"], label="DQN")
    if ddqn_df is not None and not ddqn_df.empty and {"episode", "mean100"} <= set(ddqn_df.columns):
        plt.plot(ddqn_df["episode"], ddqn_df["mean100"], label="Double DQN")
    plt.title("Training: Mean Score Over Last 100 Episodes (mean100)")
    plt.xlabel("Episode")
    plt.ylabel("Mean100")
    plt.legend()
    p2 = _save_or_show("train_mean100.png", out_dir, show, dpi)

    return [p1, p2]


def plot_eval_distributions(dqn_eval: pd.DataFrame, ddqn_eval: pd.DataFrame,
                            out_dir: Path, show: bool, dpi: int):
    paths = []

    # 3) Eval score histogram
    plt.figure()
    has_any = False
    if dqn_eval is not None and not dqn_eval.empty and "score" in dqn_eval.columns:
        plt.hist(dqn_eval["score"].astype(float), bins=20, alpha=0.6, label="DQN")
        has_any = True
    if ddqn_eval is not None and not ddqn_eval.empty and "score" in ddqn_eval.columns:
        plt.hist(ddqn_eval["score"].astype(float), bins=20, alpha=0.6, label="Double DQN")
        has_any = True
    if has_any:
        plt.title("Evaluation: Score Distribution (ε = 0)")
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.legend()
        paths.append(_save_or_show("eval_score_hist.png", out_dir, show, dpi))
    else:
        plt.close()

    # 4) Eval frames_alive histogram
    plt.figure()
    has_any = False
    if dqn_eval is not None and not dqn_eval.empty and "frames_alive" in dqn_eval.columns:
        plt.hist(dqn_eval["frames_alive"].astype(float), bins=20, alpha=0.6, label="DQN")
        has_any = True
    if ddqn_eval is not None and not ddqn_eval.empty and "frames_alive" in ddqn_eval.columns:
        plt.hist(ddqn_eval["frames_alive"].astype(float), bins=20, alpha=0.6, label="Double DQN")
        has_any = True
    if has_any:
        plt.title("Evaluation: Frames Alive Distribution (ε = 0)")
        plt.xlabel("Frames Alive")
        plt.ylabel("Count")
        plt.legend()
        paths.append(_save_or_show("eval_frames_alive_hist.png", out_dir, show, dpi))
    else:
        plt.close()

    return paths


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)

    dqn_dir = runs_dir / "dqn" / args.run_name
    ddqn_dir = runs_dir / "double_dqn" / args.run_name

    dqn_train = _safe_read_csv(dqn_dir / "metrics.csv")
    ddqn_train = _safe_read_csv(ddqn_dir / "metrics.csv")

    dqn_eval = _safe_read_csv(dqn_dir / "eval.csv")
    ddqn_eval = _safe_read_csv(ddqn_dir / "eval.csv")

    print("\n===== TRAINING SUMMARY =====")
    print(summarize_training(dqn_train, "DQN"))
    print(summarize_training(ddqn_train, "Double DQN"))

    print("\n===== EVALUATION SUMMARY (ε = 0) =====")
    print(summarize_eval(dqn_eval, "DQN"))
    print(summarize_eval(ddqn_eval, "Double DQN"))

    out_dir = runs_dir / f"compare_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    saved += plot_training(dqn_train, ddqn_train, out_dir, args.show, args.dpi)
    saved += plot_eval_distributions(dqn_eval, ddqn_eval, out_dir, args.show, args.dpi)

    print("\n===== SAVED FIGURES =====")
    if saved:
        for p in saved:
            print(f"- {p}")
    else:
        print("(No figures saved — missing CSV inputs.)")
    print()


if __name__ == "__main__":
    main()
