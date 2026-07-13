"""
卒業論文：リターン分布形状のレジームダイナミクス分析
Step 1: EWMA時変モーメント推定（日経225専用・統合版）

========================================
入力:
  - nikkei_returns.csv（Step 0で作成）

本スクリプトの内容:
  1. EWMA時変歪度・超過尖度・分散の計算
     - λ = {0.90, 0.94, 0.97, 0.99}
     - ウォームアップ期間による安定した初期化

  2. Rolling窓でのサンプル歪度・尖度（比較用）
     - 窓 = 60日, 120日

  3. 追加分析
     - 定常性検定（ADF）
     - 歪度・尖度の関係分析
     - λ選択評価

  4. 可視化（卒論第3章用・PDF＋PNG）
     - 図3-3: 日次リターンとEWMAボラティリティ（2段パネル）
     - 図3-4: 時変歪度 S_t の時系列
     - 図3-5: 時変超過尖度 ExKurt_t の時系列
     - EWMA Vol / Skew / ExKurt の3段パネル図
     - EWMA Vol 単独図
     - λ感応度（歪度・超過尖度）
     - EWMA vs Rolling比較
     - EWMA歪度・尖度とイベントタイムライン（イベント名付き）
     - 歪度・尖度の関係（散布図＋ローリング相関＋時系列）

  5. 表（数値出力）
     - 表3-3: EWMA Volatility / Skewness / Excess Kurtosis の記述統計
     - 表3-4: EWMA Volatility / Skewness / Excess Kurtosis の相関係数行列
       → LaTeX tabular を標準出力に表示

出力ファイル:
  - ewma_moments_nikkei.csv
  - ewma_moments_all_lambda.csv
  - ewma_stationarity.csv
  - ewma_lambda_evaluation.csv
  - ewma_summary_stats.csv
  - ewma_corr_matrix.csv

  - fig_return_ewmavol_2panel.pdf / .png
  - fig_skew_timeseries.pdf / .png
  - fig_kurt_timeseries.pdf / .png
  - fig_vol_skew_kurt_3panel.pdf / .png
  - fig_ewma_variance.pdf / .png
  - fig_ewma_lambda_sensitivity.pdf / .png
  - fig_ewma_vs_rolling.pdf / .png
  - fig_ewma_event_timeline.pdf / .png
  - fig_skew_kurt_relationship.pdf / .png
========================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, pearsonr
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

print("=" * 70)
print("Step 1: EWMA時変モーメント推定（日経225専用・統合版）")
print("=" * 70)


# ============================================================================
# 1. パラメータ設定 & 卒論全体で統一して用いるイベント一覧
# ============================================================================

EWMA_LAMBDAS = [0.90, 0.94, 0.97, 0.99]
LAMBDA_BASE = 0.97           # 本文で主に使う λ
WARMUP_PERIOD = 60           # ウォームアップ期間（日数）
ROLLING_WINDOWS = [60, 120]  # Rolling窓

INPUT_FILE = "nikkei_returns.csv"



# ============================================================================
# 2. データ読み込み
# ============================================================================

def load_returns_data(path: str = INPUT_FILE) -> pd.Series:
    """Step 0 で作成した日経225リターンデータを読み込み"""
    print(f"\n[データ読み込み] {path}")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  ✅ 読み込み成功")
        print(f"  期間: {df.index[0].date()} ～ {df.index[-1].date()}")
        print(f"  行数: {len(df):,}")
        if 'Return' in df.columns:
            returns = df['Return']
        elif 'Nikkei225_return' in df.columns:
            returns = df['Nikkei225_return']
        else:
            returns = df.iloc[:, 0]
        returns.name = 'Return'
        return returns

    except FileNotFoundError:
        print(f"  ⚠️ ファイルが見つかりません: {path}")
        print("  → サンプルデータを生成します（デモ用、論文本番では Step 0 を実行）")

        np.random.seed(42)
        dates = pd.date_range('2000-01-01', '2024-12-31', freq='B')
        n = len(dates)

        regime = np.zeros(n, dtype=int)
        current_regime = 0
        for i in range(1, n):
            if current_regime == 0:  # Normal
                if np.random.random() < 0.01:
                    current_regime = 1
            elif current_regime == 1:  # Stress
                if np.random.random() < 0.05:
                    current_regime = 2
                elif np.random.random() < 0.10:
                    current_regime = 0
            else:  # Crisis
                if np.random.random() < 0.15:
                    current_regime = 1
            regime[i] = current_regime

        returns = np.where(
            regime == 0, np.random.normal(0.0005, 0.012, n),
            np.where(regime == 1, np.random.normal(-0.001, 0.020, n),
                     np.random.normal(-0.005, 0.035, n))
        )
        returns = pd.Series(returns, index=dates, name='Return')
        returns.to_csv(path)
        print(f"  ✅ サンプルデータを生成・保存: {len(returns):,}行")
        return returns


# ============================================================================
# 3. EWMA時変モーメント計算
# ============================================================================

def compute_ewma_moments(
    returns: pd.Series,
    lam: float,
    warmup: int = 60
) -> pd.DataFrame:
    """
    EWMA による時変平均・分散・歪度・超過尖度（Excess Kurtosis）を計算
    """
    r = returns.dropna().values
    n = len(r)
    dates = returns.dropna().index

    if n < warmup:
        print(f"    ⚠️ データ数({n}) < ウォームアップ期間({warmup}) → 短縮します")
        warmup = max(20, n // 4)

    ewma_skew = np.full(n, np.nan)
    ewma_kurt = np.full(n, np.nan)
    ewma_var  = np.full(n, np.nan)

    warmup_data = r[:warmup]
    m1 = np.mean(warmup_data)
    m2 = np.mean(warmup_data ** 2)
    m3 = np.mean(warmup_data ** 3)
    m4 = np.mean(warmup_data ** 4)

    for i in range(warmup, n):
        x = r[i]
        m1 = lam * m1 + (1 - lam) * x
        m2 = lam * m2 + (1 - lam) * x ** 2
        m3 = lam * m3 + (1 - lam) * x ** 3
        m4 = lam * m4 + (1 - lam) * x ** 4

        mu = m1
        mu2 = m2 - mu ** 2
        if mu2 <= 1e-10:
            continue

        mu3 = m3 - 3 * mu * m2 + 2 * mu ** 3
        mu4 = m4 - 4 * mu * m3 + 6 * mu ** 2 * m2 - 3 * mu ** 4

        sigma2 = mu2
        sigma = np.sqrt(sigma2)

        ewma_var[i] = sigma2
        ewma_skew[i] = mu3 / (sigma ** 3)
        ewma_kurt[i] = mu4 / (sigma2 ** 2) - 3.0   # Excess Kurtosis

        ewma_skew[i] = np.clip(ewma_skew[i], -10, 10)
        ewma_kurt[i] = np.clip(ewma_kurt[i], -5, 50)

    lam_str = f"{lam:.2f}".replace(".", "")
    df = pd.DataFrame({
        f'skew_ewma_{lam_str}': ewma_skew,
        f'kurt_ewma_{lam_str}': ewma_kurt,
        f'var_ewma_{lam_str}': ewma_var,
    }, index=dates)
    return df


def compute_all_ewma(returns: pd.Series, lambdas: list, warmup: int = 60) -> pd.DataFrame:
    print(f"\n[EWMA計算]")
    dfs = []
    for lam in lambdas:
        df = compute_ewma_moments(returns, lam, warmup)
        dfs.append(df)
        print(f"  λ={lam:.2f} ✓")
    return pd.concat(dfs, axis=1)


# ============================================================================
# 4. Rolling窓での歪度・尖度計算
# ============================================================================

def compute_rolling_moments(
    returns: pd.Series,
    windows: list,
    min_frac: float = 0.5
) -> pd.DataFrame:
    """Rolling窓によるサンプル歪度・超過尖度"""
    r = returns.copy()
    df_out = pd.DataFrame(index=r.index)
    print(f"\n[Rolling計算]")
    for w in windows:
        min_periods = max(1, int(w * min_frac))
        df_out[f'skew_roll_{w}'] = r.rolling(
            window=w, min_periods=min_periods
        ).apply(lambda x: skew(x, bias=False), raw=False)
        df_out[f'kurt_roll_{w}'] = r.rolling(
            window=w, min_periods=min_periods
        ).apply(lambda x: kurtosis(x, fisher=True, bias=False), raw=False)
        print(f"  窓={w}日 ✓")
    return df_out


# ============================================================================
# 5. 定常性検定
# ============================================================================

def test_stationarity(series: pd.Series, name: str) -> dict:
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'Variable': name,
        'ADF_Statistic': result[0],
        'p_value': result[1],
        'Lags_Used': result[2],
        'Critical_1%': result[4]['1%'],
        'Critical_5%': result[4]['5%'],
        'Is_Stationary': result[1] < 0.05,
    }


def run_stationarity_tests(df: pd.DataFrame, lam: float = LAMBDA_BASE) -> pd.DataFrame:
    print("\n[定常性検定 (ADF)]")
    lam_str = f"{lam:.2f}".replace(".", "")
    test_cols = [
        (f'skew_ewma_{lam_str}', '歪度'),
        (f'kurt_ewma_{lam_str}', '超過尖度'),
    ]
    results = []
    for col, name in test_cols:
        if col in df.columns:
            result = test_stationarity(df[col], name)
            results.append(result)
            status = "✓ 定常" if result['Is_Stationary'] else "✗ 非定常"
            print(f"  {name}: ADF={result['ADF_Statistic']:.3f}, p={result['p_value']:.4f} {status}")
    return pd.DataFrame(results)


# ============================================================================
# 6. 歪度・尖度の関係分析
# ============================================================================

def analyze_skew_kurt_relationship(df: pd.DataFrame, lam: float = LAMBDA_BASE) -> dict:
    print("\n[歪度・尖度の関係分析]")
    lam_str = f"{lam:.2f}".replace(".", "")
    skew_col = f'skew_ewma_{lam_str}'
    kurt_col = f'kurt_ewma_{lam_str}'

    skew_series = df[skew_col].dropna()
    kurt_series = df[kurt_col].dropna()
    common_idx = skew_series.index.intersection(kurt_series.index)
    skew_common = skew_series.loc[common_idx]
    kurt_common = kurt_series.loc[common_idx]

    corr, p_val = pearsonr(skew_common, kurt_common)
    print(f"  歪度-尖度相関: {corr:.4f} (p={p_val:.2e})")

    rolling_corr_60 = skew_common.rolling(60).corr(kurt_common)
    rolling_corr_250 = skew_common.rolling(250).corr(kurt_common)

    stats = {
        'skew_mean': skew_common.mean(),
        'skew_std': skew_common.std(),
        'kurt_mean': kurt_common.mean(),
        'kurt_std': kurt_common.std(),
        'correlation': corr,
        'correlation_pval': p_val,
    }

    skew_q33, skew_q67 = skew_common.quantile([0.33, 0.67])
    kurt_q33, kurt_q67 = kurt_common.quantile([0.33, 0.67])

    print("\n  [レジーム予備分類に使える分位点]")
    print(f"    歪度 33/67パーセンタイル: {skew_q33:.3f} / {skew_q67:.3f}")
    print(f"    尖度 33/67パーセンタイル: {kurt_q33:.3f} / {kurt_q67:.3f}")

    return {
        'stats': stats,
        'rolling_corr_60': rolling_corr_60,
        'rolling_corr_250': rolling_corr_250,
        'skew': skew_common,
        'kurt': kurt_common,
        'quantiles': {
            'skew_q33': skew_q33, 'skew_q67': skew_q67,
            'kurt_q33': kurt_q33, 'kurt_q67': kurt_q67,
        }
    }


# ============================================================================
# 7. λ選択評価
# ============================================================================

def evaluate_lambda_selection(
    returns: pd.Series,
    lambdas: list,
    warmup: int = 60
) -> pd.DataFrame:
    print("\n[λ選択評価]")
    results = []
    for lam in lambdas:
        df = compute_ewma_moments(returns, lam, warmup)
        lam_str = f"{lam:.2f}".replace(".", "")
        skew_col = f'skew_ewma_{lam_str}'
        kurt_col = f'kurt_ewma_{lam_str}'
        skew_series = df[skew_col].dropna()
        kurt_series = df[kurt_col].dropna()
        results.append({
            'Lambda': lam,
            'Skew_Std': skew_series.std(),
            'Kurt_Std': kurt_series.std(),
            'Skew_AC1': skew_series.autocorr(lag=1),
            'Kurt_AC1': kurt_series.autocorr(lag=1),
            'Skew_Change': skew_series.diff().abs().mean(),
            'Kurt_Change': kurt_series.diff().abs().mean(),
        })
    df_eval = pd.DataFrame(results)
    print(df_eval.to_string(index=False))
    return df_eval


# ============================================================================
# 8. 要約統計・相関（表3-3 & 表3-4）
# ============================================================================

def compute_ewma_summary_and_corr(df: pd.DataFrame, lam: float = LAMBDA_BASE):
    """
    表3-3: EWMA Vol / Skew / ExKurt の記述統計
    表3-4: EWMA Vol / Skew / ExKurt の相関係数行列
    を計算し、CSV保存＋LaTeX tabular を print。
    """
    lam_str = f"{lam:.2f}".replace(".", "")
    col_var = f'var_ewma_{lam_str}'
    col_skew = f'skew_ewma_{lam_str}'
    col_kurt = f'kurt_ewma_{lam_str}'

    if not all(c in df.columns for c in [col_var, col_skew, col_kurt]):
        print("⚠️ 必要なEWMA列が不足しています。")
        return None, None

    vol = np.sqrt(df[col_var]) * 100.0  # パーセント表示
    skew_series = df[col_skew]
    kurt_series = df[col_kurt]

    data = pd.concat([
        vol.rename('Volatility'),
        skew_series.rename('Skewness'),
        kurt_series.rename('ExcessKurtosis')
    ], axis=1).dropna()

    summary = pd.DataFrame({
        'Mean': data.mean(),
        'Std': data.std(),
        'Min': data.min(),
        'Max': data.max(),
    })

    corr_mat = data.corr()

    summary.to_csv('ewma_summary_stats.csv')
    corr_mat.to_csv('ewma_corr_matrix.csv')
    print("✅ ewma_summary_stats.csv（表3-3用）")
    print("✅ ewma_corr_matrix.csv（表3-4用）")

    print("\n===== 表3-3: EWMA Vol / Skew / ExKurt 記述統計 (LaTeX) =====")
    print(summary.to_latex(float_format="%.4f"))

    print("\n===== 表3-4: EWMA Vol / Skew / ExKurt 相関行列 (LaTeX) =====")
    print(corr_mat.to_latex(float_format="%.4f"))

    return summary, corr_mat


# ============================================================================
# 9. 可視化用ユーティリティ（EVENTS を用いてイベント名付き）
# ============================================================================

def add_event_lines(ax, index, alpha=0.5, with_labels=False, label_pos=0.95):
    """
    卒論全体で統一した EVENTS に従って主要イベントに縦線を引く。
    - グローバル: 赤・破線
    - 日本     : 青・点線
    with_labels=True のときはイベント名も描画。
    """
    start, end = index[0], index[-1]

    for ev in EVENTS:
        dt = pd.Timestamp(ev["date"])
        if not (start <= dt <= end):
            continue

        if ev["region"] == "global":
            color = "red"
            ls = "--"
        else:
            color = "blue"
            ls = ":"

        ax.axvline(dt, color=color, ls=ls, alpha=alpha, lw=1.2)

        if with_labels:
            ymin, ymax = ax.get_ylim()
            if ev["region"] == "global":
                y = ymax * label_pos
                va = "top"
            else:
                y = ymin * (2 - label_pos)
                va = "bottom"
            ax.text(
                dt, y, ev["name"],
                fontsize=7, rotation=90,
                ha="right", va=va, color=color
            )


# ============================================================================
# 10. 可視化: 図3-3〜図3-5 & 3段パネルなど
# ============================================================================

def plot_return_and_volatility_2panel(df: pd.DataFrame,
                                      lam: float,
                                      save_pdf: str = "fig_return_ewmavol_2panel.pdf",
                                      save_png: str = "fig_return_ewmavol_2panel.png"):
    """
    図3-3: 日次リターンと EWMA ボラティリティの時系列（2段パネル）
    上: Return, 下: EWMA Volatility
    """
    lam_str = f"{lam:.2f}".replace(".", "")
    col_var = f'var_ewma_{lam_str}'
    if col_var not in df.columns or 'Return' not in df.columns:
        print("⚠️ Return または EWMA variance がありません。")
        return

    vol = np.sqrt(df[col_var]) * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # 上: リターン
    ax = axes[0]
    ax.plot(df.index, df['Return'] * 100, lw=0.7, color='black')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title("日次リターンの推移", fontweight='bold')
    ax.set_ylabel("日次リターン（％）")
    add_event_lines(ax, df.index, alpha=0.4, with_labels=True, label_pos=0.95)

    # 下: EWMAボラ
    ax = axes[1]
    ax.plot(df.index, vol, lw=1, color='darkgreen')
    ax.set_title(f"EWMAボラティリティの推移 (λ={lam:.2f})", fontweight='bold')
    ax.set_ylabel("ボラティリティ（％）")
    ax.set_xlabel("日付")
    add_event_lines(ax, df.index, alpha=0.4, with_labels=True, label_pos=0.95)

    plt.tight_layout()
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {save_pdf}")
    if save_png is not None:
        plt.savefig(save_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 保存: {save_png}")
    plt.show()
    plt.close()


def plot_skew_timeseries(df: pd.DataFrame,
                         lam: float,
                         save_pdf: str = "fig_skew_timeseries.pdf",
                         save_png: str = "fig_skew_timeseries.png"):
    """
    図3-4: 時変歪度 S_t の時系列
    """
    lam_str = f"{lam:.2f}".replace(".", "")
    col_skew = f'skew_ewma_{lam_str}'
    if col_skew not in df.columns:
        print("⚠️ EWMA 歪度列がありません。")
        return

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(df.index, df[col_skew], lw=1, color='navy')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title(f"時変歪度 $S_t$ の時系列 (EWMA, λ={lam:.2f})", fontweight='bold')
    ax.set_ylabel("歪度")
    ax.set_xlabel("日付")
    add_event_lines(ax, df.index, alpha=0.4, with_labels=True, label_pos=0.95)

    plt.tight_layout()
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {save_pdf}")
    if save_png is not None:
        plt.savefig(save_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 保存: {save_png}")
    plt.show()
    plt.close()


def plot_kurt_timeseries(df: pd.DataFrame,
                         lam: float,
                         save_pdf: str = "fig_kurt_timeseries.pdf",
                         save_png: str = "fig_kurt_timeseries.png"):
    """
    図3-5: 時変超過尖度 ExKurt_t の時系列
    """
    lam_str = f"{lam:.2f}".replace(".", "")
    col_kurt = f'kurt_ewma_{lam_str}'
    if col_kurt not in df.columns:
        print("⚠️ EWMA 超過尖度列がありません。")
        return

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(df.index, df[col_kurt], lw=1, color='darkred')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title(f"時変超過尖度 $ExKurt_t$ の時系列 (EWMA, λ={lam:.2f})", fontweight='bold')
    ax.set_ylabel("超過尖度")
    ax.set_xlabel("日付")
    add_event_lines(ax, df.index, alpha=0.4, with_labels=True, label_pos=0.95)

    plt.tight_layout()
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {save_pdf}")
    if save_png is not None:
        plt.savefig(save_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 保存: {save_png}")
    plt.show()
    plt.close()


def plot_vol_skew_kurt_3panel(df: pd.DataFrame,
                              lam: float,
                              save_pdf: str = "fig_vol_skew_kurt_3panel.pdf",
                              save_png: str = "fig_vol_skew_kurt_3panel.png"):
    """
    EWMA Volatility / Skewness / Excess Kurtosis の3段パネル図
    """
    lam_str = f"{lam:.2f}".replace(".", "")
    col_var = f'var_ewma_{lam_str}'
    col_skew = f'skew_ewma_{lam_str}'
    col_kurt = f'kurt_ewma_{lam_str}'

    if not all(c in df.columns for c in [col_var, col_skew, col_kurt]):
        print("⚠️ EWMA 列が不足しています。")
        return

    vol = np.sqrt(df[col_var]) * 100.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    ax.plot(df.index, vol, lw=1, color='darkgreen')
    ax.set_title(f"EWMAボラティリティ (λ={lam:.2f})", fontweight='bold')
    ax.set_ylabel("Vol（％）")
    add_event_lines(ax, df.index, alpha=0.4, with_labels=True, label_pos=0.95)

    ax = axes[1]
    ax.plot(df.index, df[col_skew], lw=1, color='navy')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title("EWMA歪度 $S_t$", fontweight='bold')
    ax.set_ylabel("歪度")
    add_event_lines(ax, df.index, alpha=0.4, with_labels=True, label_pos=0.95)

    ax = axes[2]
    ax.plot(df.index, df[col_kurt], lw=1, color='darkred')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title("EWMA超過尖度 $ExKurt_t$", fontweight='bold')
    ax.set_ylabel("超過尖度")
    ax.set_xlabel("日付")
    add_event_lines(ax, df.index, alpha=0.4, with_labels=True, label_pos=0.95)

    plt.tight_layout()
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {save_pdf}")
    if save_png is not None:
        plt.savefig(save_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 保存: {save_png}")
    plt.show()
    plt.close()


def plot_ewma_variance(df: pd.DataFrame,
                       lam: float,
                       save_path_pdf: str = "fig_ewma_variance.pdf",
                       save_path_png: str = "fig_ewma_variance.png") -> None:
    """EWMAボラティリティ単独の図"""
    lam_str = f"{lam:.2f}".replace(".", "")
    col_var = f'var_ewma_{lam_str}'
    if col_var not in df.columns:
        print(f"⚠️ {col_var} が見つかりません")
        return
    vol = np.sqrt(df[col_var]) * 100
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, vol, lw=1, color="darkgreen")
    ax.set_title(f"日経225 EWMAボラティリティの推移 (λ={lam:.2f})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("ボラティリティ（％）")
    ax.set_xlabel("日付")
    add_event_lines(ax, df.index, alpha=0.4, with_labels=True, label_pos=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ 保存: {save_path_pdf}")
    if save_path_png is not None:
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✅ 保存: {save_path_png}")
    plt.show()
    plt.close()


def plot_lambda_sensitivity(df: pd.DataFrame,
                            lambdas: list,
                            save_path_pdf: str = 'fig_ewma_lambda_sensitivity.pdf',
                            save_path_png: str = 'fig_ewma_lambda_sensitivity.png') -> None:
    """λ感応度図（歪度・超過尖度）"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(lambdas)))

    ax = axes[0]
    for i, lam in enumerate(lambdas):
        lam_str = f"{lam:.2f}".replace(".", "")
        col = f'skew_ewma_{lam_str}'
        if col in df.columns:
            ax.plot(df.index, df[col], lw=1, color=colors[i], label=f'λ={lam:.2f}')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title('日経225: EWMA歪度 (λ感応度)', fontsize=12, fontweight='bold')
    ax.set_ylabel('歪度')
    ax.legend(loc='upper right', fontsize=9)
    add_event_lines(ax, df.index, alpha=0.2, with_labels=False)

    ax = axes[1]
    for i, lam in enumerate(lambdas):
        lam_str = f"{lam:.2f}".replace(".", "")
        col = f'kurt_ewma_{lam_str}'
        if col in df.columns:
            ax.plot(df.index, df[col], lw=1, color=colors[i], label=f'λ={lam:.2f}')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title('日経225: EWMA超過尖度 (λ感応度)', fontsize=12, fontweight='bold')
    ax.set_ylabel('超過尖度')
    ax.set_xlabel('日付')
    ax.legend(loc='upper right', fontsize=9)
    add_event_lines(ax, df.index, alpha=0.2, with_labels=False)

    plt.tight_layout()
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {save_path_pdf}")
    if save_path_png is not None:
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 保存: {save_path_png}")
    plt.show()
    plt.close()


def plot_ewma_vs_rolling(df: pd.DataFrame,
                         lam: float,
                         windows: list,
                         save_path_pdf: str = 'fig_ewma_vs_rolling.pdf',
                         save_path_png: str = 'fig_ewma_vs_rolling.png') -> None:
    """EWMA vs Rolling の比較図（歪度・超過尖度）"""
    lam_str = f"{lam:.2f}".replace(".", "")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    col_ewma = f'skew_ewma_{lam_str}'
    if col_ewma in df.columns:
        ax.plot(df.index, df[col_ewma], lw=1.5, color='navy', label=f'EWMA λ={lam:.2f}')
    for w in windows:
        col_roll = f'skew_roll_{w}'
        if col_roll in df.columns:
            ax.plot(df.index, df[col_roll], lw=1, alpha=0.7, label=f'Rolling {w}日')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title('日経225: 歪度 (EWMA vs Rolling)', fontsize=12, fontweight='bold')
    ax.set_ylabel('歪度')
    ax.legend(loc='upper right', fontsize=9)
    add_event_lines(ax, df.index, alpha=0.2, with_labels=False)

    ax = axes[1]
    col_ewma = f'kurt_ewma_{lam_str}'
    if col_ewma in df.columns:
        ax.plot(df.index, df[col_ewma], lw=1.5, color='darkred', label=f'EWMA λ={lam:.2f}')
    for w in windows:
        col_roll = f'kurt_roll_{w}'
        if col_roll in df.columns:
            ax.plot(df.index, df[col_roll], lw=1, alpha=0.7, label=f'Rolling {w}日')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title('日経225: 超過尖度 (EWMA vs Rolling)', fontsize=12, fontweight='bold')
    ax.set_ylabel('超過尖度')
    ax.set_xlabel('日付')
    ax.legend(loc='upper right', fontsize=9)
    add_event_lines(ax, df.index, alpha=0.2, with_labels=False)

    plt.tight_layout()
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {save_path_pdf}")
    if save_path_png is not None:
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 保存: {save_path_png}")
    plt.show()
    plt.close()


def plot_event_timeline(df: pd.DataFrame,
                        lam: float,
                        save_path_pdf: str = 'fig_ewma_event_timeline.pdf',
                        save_path_png: str = 'fig_ewma_event_timeline.png') -> None:
    """EWMA歪度・超過尖度とイベント名を明示したタイムライン図"""
    lam_str = f"{lam:.2f}".replace(".", "")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax = axes[0]
    col = f'skew_ewma_{lam_str}'
    if col in df.columns:
        ax.plot(df.index, df[col], lw=1, color='navy')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title(f'日経225 EWMA歪度と主要イベント (λ={lam:.2f})', fontsize=12, fontweight='bold')
    ax.set_ylabel('歪度')
    add_event_lines(ax, df.index, alpha=0.8, with_labels=True, label_pos=0.95)

    ax = axes[1]
    col = f'kurt_ewma_{lam_str}'
    if col in df.columns:
        ax.plot(df.index, df[col], lw=1, color='darkred')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title(f'日経225 EWMA超過尖度と主要イベント (λ={lam:.2f})', fontsize=12, fontweight='bold')
    ax.set_ylabel('超過尖度')
    ax.set_xlabel('日付')
    add_event_lines(ax, df.index, alpha=0.8, with_labels=True, label_pos=0.95)

    plt.tight_layout()
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {save_path_pdf}")
    if save_path_png is not None:
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 保存: {save_path_png}")
    plt.show()
    plt.close()


def plot_skew_kurt_relationship_fig(
    analysis: dict,
    save_path_pdf: str = 'fig_skew_kurt_relationship.pdf',
    save_path_png: str = 'fig_skew_kurt_relationship.png'
) -> None:
    """歪度・尖度の関係可視化（散布図＋ローリング相関＋時系列）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    skew_series = analysis['skew']
    kurt_series = analysis['kurt']
    stats = analysis['stats']
    q = analysis['quantiles']

    ax = axes[0, 0]
    ax.scatter(skew_series, kurt_series, alpha=0.2, s=5, c='steelblue')
    z = np.polyfit(skew_series, kurt_series, 1)
    p = np.poly1d(z)
    x_line = np.linspace(skew_series.min(), skew_series.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', lw=2, label='回帰直線')
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('歪度')
    ax.set_ylabel('超過尖度')
    ax.set_title(f'(a) 歪度 vs 超過尖度 (ρ={stats["correlation"]:.3f})', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(analysis['rolling_corr_60'].index, analysis['rolling_corr_60'],
            lw=1, alpha=0.7, label='60日')
    ax.plot(analysis['rolling_corr_250'].index, analysis['rolling_corr_250'],
            lw=1.5, label='250日')
    ax.axhline(0, color='black', lw=0.5)
    ax.axhline(stats['correlation'], color='red', ls='--', lw=1,
               label=f'全期間平均: {stats["correlation"]:.2f}')
    ax.set_ylabel('相関係数')
    ax.set_xlabel('日付')
    ax.set_title('(b) ローリング相関（歪度-尖度）', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(-1, 1)
    add_event_lines(ax, skew_series.index, alpha=0.25, with_labels=True, label_pos=0.95)

    ax = axes[1, 0]
    ax.plot(skew_series.index, skew_series, lw=0.8, color='navy')
    ax.axhline(0, color='black', lw=0.5)
    ax.axhline(q['skew_q33'], color='orange', ls=':', alpha=0.7, label='33%ile')
    ax.axhline(q['skew_q67'], color='green', ls=':', alpha=0.7, label='67%ile')
    ax.set_ylabel('歪度')
    ax.set_xlabel('日付')
    ax.set_title('(c) 歪度の時系列', fontweight='bold')
    ax.legend(loc='upper right')
    add_event_lines(ax, skew_series.index, alpha=0.25, with_labels=True, label_pos=0.95)

    ax = axes[1, 1]
    ax.plot(kurt_series.index, kurt_series, lw=0.8, color='darkred')
    ax.axhline(0, color='black', lw=0.5)
    ax.axhline(q['kurt_q33'], color='orange', ls=':', alpha=0.7, label='33%ile')
    ax.axhline(q['kurt_q67'], color='green', ls=':', alpha=0.7, label='67%ile')
    ax.set_ylabel('超過尖度')
    ax.set_xlabel('日付')
    ax.set_title('(d) 超過尖度の時系列', fontweight='bold')
    ax.legend(loc='upper right')
    add_event_lines(ax, kurt_series.index, alpha=0.25, with_labels=True, label_pos=0.95)

    plt.suptitle('歪度・超過尖度の関係分析（第3章 3.5節）',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {save_path_pdf}")
    if save_path_png is not None:
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 保存: {save_path_png}")
    plt.show()
    plt.close()


# ============================================================================
# 11. メイン実行
# ============================================================================

def main():
    # 1. データ読み込み
    returns = load_returns_data(INPUT_FILE)
    if returns is None:
        print("\n❌ データが読み込めません。Step 0を先に実行してください。")
        return None

    # 2. EWMA時変モーメント
    print("\n" + "=" * 70)
    print("EWMA時変モーメント計算")
    print("=" * 70)
    df_ewma = compute_all_ewma(returns, EWMA_LAMBDAS, WARMUP_PERIOD)

    # 3. Rollingモーメント
    print("\n" + "=" * 70)
    print("Rolling窓モーメント計算")
    print("=" * 70)
    df_roll = compute_rolling_moments(returns, ROLLING_WINDOWS)

    # 結合
    df = pd.concat([returns, df_ewma, df_roll], axis=1)

    # 4. 定常性検定
    print("\n" + "=" * 70)
    print("定常性検定")
    print("=" * 70)
    stationarity_results = run_stationarity_tests(df, LAMBDA_BASE)

    # 5. 歪度・尖度の関係分析
    print("\n" + "=" * 70)
    print("歪度・尖度の関係分析（3.5節）")
    print("=" * 70)
    skew_kurt_analysis = analyze_skew_kurt_relationship(df, LAMBDA_BASE)

    # 6. λ選択評価
    print("\n" + "=" * 70)
    print("λ選択評価")
    print("=" * 70)
    lambda_eval = evaluate_lambda_selection(returns, EWMA_LAMBDAS, WARMUP_PERIOD)

    # 7. 要約統計・相関（表3-3 / 3-4）
    print("\n" + "=" * 70)
    print("要約統計・相関（表3-3 / 表3-4）")
    print("=" * 70)
    summary_stats, corr_mat = compute_ewma_summary_and_corr(df, LAMBDA_BASE)

    # 8. データ保存
    print("\n" + "=" * 70)
    print("データ保存")
    print("=" * 70)
    lam_str = f"{LAMBDA_BASE:.2f}".replace(".", "")
    df_for_ms = df[[
        'Return',
        f'skew_ewma_{lam_str}',
        f'kurt_ewma_{lam_str}',
        f'var_ewma_{lam_str}',
    ]].copy()
    df_for_ms.columns = ['Return', 'Skewness', 'ExcessKurtosis', 'Variance']
    df_for_ms.to_csv('ewma_moments_nikkei.csv')
    print("✅ ewma_moments_nikkei.csv（Step 2: MSモデルへの入力）")

    df.to_csv('ewma_moments_all_lambda.csv')
    print("✅ ewma_moments_all_lambda.csv（全λのデータ）")

    stationarity_results.to_csv('ewma_stationarity.csv', index=False)
    print("✅ ewma_stationarity.csv")

    lambda_eval.to_csv('ewma_lambda_evaluation.csv', index=False)
    print("✅ ewma_lambda_evaluation.csv")

    # 9. 可視化（第3章用）
    print("\n" + "=" * 70)
    print("可視化（卒論第3章用図表）")
    print("=" * 70)

    # 図3-3
    plot_return_and_volatility_2panel(df, LAMBDA_BASE)

    # 図3-4
    plot_skew_timeseries(df, LAMBDA_BASE)

    # 図3-5
    plot_kurt_timeseries(df, LAMBDA_BASE)

    # Vol / Skew / Kurt 3段パネル
    plot_vol_skew_kurt_3panel(df, LAMBDA_BASE)

    # EWMA Vol 単独
    plot_ewma_variance(df, LAMBDA_BASE)

    # λ感応度
    plot_lambda_sensitivity(df, EWMA_LAMBDAS)

    # EWMA vs Rolling
    plot_ewma_vs_rolling(df, LAMBDA_BASE, ROLLING_WINDOWS)

    # EWMA歪度・尖度とイベント名
    plot_event_timeline(df, LAMBDA_BASE)

    # 歪度・尖度の関係図
    plot_skew_kurt_relationship_fig(skew_kurt_analysis)

    # サマリー
    print("\n" + "=" * 70)
    print("Step 1 完了サマリー")
    print("=" * 70)

    if summary_stats is not None:
        stats = skew_kurt_analysis['stats']
        print(f"""
✅ EWMA時変モーメント計算完了
   - λ 候補 = {EWMA_LAMBDAS}
   - 本文で採用する λ = {LAMBDA_BASE}
   - ウォームアップ期間 = {WARMUP_PERIOD}日

✅ 定常性検定（ewma_stationarity.csv）
✅ λ選択評価（ewma_lambda_evaluation.csv）
✅ 表3-3 / 表3-4 用データ
   - ewma_summary_stats.csv
   - ewma_corr_matrix.csv
   - LaTeX tabular はコンソール出力済み

✅ 歪度・尖度の関係（3.5節）
   - 歪度: 平均={stats['skew_mean']:.4f}, 標準偏差={stats['skew_std']:.4f}
   - 超過尖度: 平均={stats['kurt_mean']:.4f}, 標準偏差={stats['kurt_std']:.4f}
   - 相関: {stats['correlation']:.4f}

→ 第4章のMSモデル推定には ewma_moments_nikkei.csv を使用
""")

    return {
        'df': df,
        'df_for_ms': df_for_ms,
        'stationarity_results': stationarity_results,
        'skew_kurt_analysis': skew_kurt_analysis,
        'lambda_eval': lambda_eval,
        'summary_stats': summary_stats,
        'corr_mat': corr_mat,
    }


if __name__ == "__main__":
    results = main()