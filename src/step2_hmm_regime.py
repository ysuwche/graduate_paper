"""
卒業論文：リターン分布形状のレジームダイナミクス分析
Step 2: 3レジーム Gaussian HMM 分析（完全統合版）- 可視化改善版

【改善点】
  - レジーム特性プロットの視認性向上
  - 観測割合と定常分布の比較を明示
  - 期待持続期間を独立したサブプロットに
  - 文字重なりの解消

【主な機能】
  1. 4つの危機因子（logVol, Skew, Kurt, SKFactor）でHMM推定
  2. 全イベントでの前後Crisis確率プロット
  3. イベント前後での各因子（Vol, Skew, Kurt, SK）の推移
  4. モデル間比較（時系列カラーマップ）
  5. 条件付きリターン分布・モーメント空間可視化
  6. LaTeX用サマリー表の出力
"""

# 依存ライブラリは requirements.txt からインストールしてください:
#   pip install -r requirements.txt
# !pip install hmmlearn

from __future__ import annotations

import copy
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from hmmlearn.hmm import GaussianHMM
from scipy import stats

# ============================================================================
# 0. matplotlib 設定
# ============================================================================

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak')


def setup_matplotlib():
    """matplotlib の日本語フォント設定"""
    import matplotlib.font_manager as fm

    plt.rcParams.update({
        'axes.unicode_minus': False,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'figure.autolayout': False,
    })

    jp_fonts = ['IPAexGothic', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'MS Gothic']
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in jp_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"  日本語フォント設定: {font}")
            return

    print("  ⚠️ 日本語フォントが見つかりません")


# ============================================================================
# 1. 設定クラスとイベント一覧
# ============================================================================

@dataclass
class Config:
    input_file: str = "ewma_moments_nikkei.csv"
    output_dir: str = "step2_outputs"

    regime_candidates: tuple[int, ...] = (2, 3, 4)
    n_states: int = 3
    n_init: int = 10
    n_iter: int = 1000
    random_state: int = 42
    convergence_tol: float = 1e-4

    regime_names: dict[int, str] = field(default_factory=lambda: {
        0: "Normal",
        1: "Stress",
        2: "Crisis",
    })

    # 色覚バリアフリー対応
    regime_colors: dict[int, str] = field(default_factory=lambda: {
        0: "#009E73",  # 緑
        1: "#E69F00",  # オレンジ
        2: "#D55E00",  # 赤
    })


# ユーザー指定のイベント一覧
EVENTS = [
    {"name": "ITバブル崩壊", "date": "2000-03-10", "region": "global"},
    {"name": "9.11テロ", "date": "2001-09-11", "region": "global"},
    {"name": "サブプライム", "date": "2007-08-09", "region": "global"},
    {"name": "リーマンショック", "date": "2008-09-15", "region": "global"},
    {"name": "東日本大震災", "date": "2011-03-11", "region": "japan"},
    {"name": "バーナンキショック", "date": "2013-05-23", "region": "global"},
    {"name": "チャイナショック", "date": "2015-08-11", "region": "global"},
    {"name": "Brexit", "date": "2016-06-24", "region": "global"},
    {"name": "トランプ当選", "date": "2016-11-09", "region": "global"},
    {"name": "クリスマスショック", "date": "2018-12-25", "region": "global"},
    {"name": "コロナショック", "date": "2020-02-24", "region": "global"},
    {"name": "ウクライナ侵攻", "date": "2022-02-24", "region": "global"},
    {"name": "令和のブラックマンデー", "date": "2024-08-05", "region": "japan"},
]


# モデル表示設定
MODEL_INFO = {
    "logvol": {"label": "Volatility", "color": "#0072B2", "linestyle": "-"},
    "skew":   {"label": "Skewness",   "color": "#009E73", "linestyle": "--"},
    "kurt":   {"label": "Kurtosis",   "color": "#CC79A7", "linestyle": "-."},
    "skfactor": {"label": "SK Factor", "color": "#D55E00", "linestyle": ":"},
}


# ============================================================================
# 2. イベント線描画（画像スタイル準拠）
# ============================================================================

def add_event_lines(ax, show_labels: bool = True, label_fontsize: int = 7):
    """
    イベント線を描画（画像のスタイルに準拠）
    - global: 赤の破線
    - japan: 青の点線
    """
    ymin, ymax = ax.get_ylim()

    for ev in EVENTS:
        dt = pd.Timestamp(ev["date"])
        region = ev.get("region", "global")

        if region == "japan":
            color = "#0000CC"  # 青
            linestyle = ":"
            alpha = 0.7
        else:
            color = "#CC0000"  # 赤
            linestyle = "--"
            alpha = 0.6

        ax.axvline(dt, color=color, ls=linestyle, alpha=alpha, lw=1.0)

        if show_labels:
            # 縦書き風にイベント名を表示
            ax.text(dt, ymax * 0.98, ev["name"],
                    fontsize=label_fontsize,
                    rotation=90, ha="right", va="top",
                    color=color, alpha=0.9)


def add_event_lines_relative(ax, event_day: int = 0, show_label: bool = True):
    """相対日数プロットでのイベント日マーカー"""
    ax.axvline(event_day, color="black", lw=1.5, ls="--")
    if show_label:
        ax.text(event_day + 1, ax.get_ylim()[1] * 0.95, "イベント日",
                fontsize=9, va="top")


# ============================================================================
# 3. データ読み込みと危機因子構築
# ============================================================================

def load_data(config: Config) -> pd.DataFrame | None:
    print("\n[データ読み込み]")
    path = Path(config.input_file)

    if not path.exists():
        print(f"  ❌ {path} が見つかりません")
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

        required_cols = ['Return', 'Skewness', 'ExcessKurtosis', 'Variance']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  ❌ 必須列が不足: {missing}")
            return None

        df = df.dropna(subset=required_cols)
        print(f"  ✅ 読み込み完了: {len(df):,} 行")
        print(f"  期間: {df.index[0].date()} ～ {df.index[-1].date()}")
        return df

    except Exception as e:
        print(f"  ❌ 読み込みエラー: {e}")
        return None


def construct_factors(df: pd.DataFrame) -> pd.DataFrame:
    """危機因子の構築"""
    print("\n[危機因子の構築]")

    variance = df["Variance"].copy()
    vol_raw = np.sqrt(np.maximum(variance.values, 0.0))
    vol_raw_safe = np.where(vol_raw <= 0, 1e-8, vol_raw)
    log_vol = np.log(vol_raw_safe)

    vol_raw = pd.Series(vol_raw, index=df.index, name="VolatilityRaw")
    log_vol = pd.Series(log_vol, index=df.index, name="LogVolatility")

    skew = df["Skewness"].copy()
    kurt = df["ExcessKurtosis"].copy()

    # SKFactor: Z(-Skewness) + Z(ExcessKurtosis) の平均
    neg_skew = -skew
    neg_skew_z = (neg_skew - neg_skew.mean()) / neg_skew.std() if neg_skew.std() > 0 else 0
    kurt_z = (kurt - kurt.mean()) / kurt.std() if kurt.std() > 0 else 0
    sk_factor = 0.5 * (neg_skew_z + kurt_z)

    df_factors = pd.DataFrame({
        "Return": df["Return"],
        "VolatilityRaw": vol_raw,
        "LogVolatility": log_vol,
        "Skewness": skew,
        "ExcessKurtosis": kurt,
        "Variance": variance,
        "SKFactor": sk_factor,
    })

    print("  基本統計量:")
    for col in ["LogVolatility", "Skewness", "ExcessKurtosis", "SKFactor"]:
        s = df_factors[col]
        print(f"    {col}: mean={s.mean():.4f}, std={s.std():.4f}")

    return df_factors


# ============================================================================
# 4. Gaussian HMM 推定
# ============================================================================

@dataclass
class HMMResult:
    model: GaussianHMM
    n_states: int
    log_likelihood: float
    aic: float
    bic: float
    posterior: pd.DataFrame
    viterbi: pd.Series
    converged: bool
    n_iter_used: int


def count_hmm_params(n_states: int) -> int:
    return n_states**2 + n_states - 1


def fit_hmm_single(X: np.ndarray, n_states: int,
                   random_state: int, n_iter: int, tol: float):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        init_params="stmc",
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)
    return model, model.score(X), model.monitor_.converged


def fit_hmm_robust(series: pd.Series, n_states: int, config: Config) -> HMMResult:
    y = series.dropna()
    X = y.values.reshape(-1, 1)
    n = len(y)

    best_model, best_ll, best_converged = None, -np.inf, False

    for i in range(config.n_init):
        model, ll, converged = fit_hmm_single(
            X, n_states, config.random_state + i, config.n_iter, config.convergence_tol
        )
        if ll > best_ll:
            best_ll, best_model, best_converged = ll, model, converged

    if best_model is None:
        raise RuntimeError(f"HMM推定に失敗")

    k = count_hmm_params(n_states)
    aic = 2 * k - 2 * best_ll
    bic = k * np.log(n) - 2 * best_ll

    posterior = best_model.predict_proba(X)
    viterbi_states = best_model.predict(X)

    return HMMResult(
        model=best_model,
        n_states=n_states,
        log_likelihood=best_ll,
        aic=aic,
        bic=bic,
        posterior=pd.DataFrame(posterior, index=y.index, columns=list(range(n_states))),
        viterbi=pd.Series(viterbi_states, index=y.index, name="Regime"),
        converged=best_converged,
        n_iter_used=best_model.monitor_.iter,
    )


def select_num_states(series: pd.Series, config: Config) -> dict[int, HMMResult]:
    print("\n  [レジーム数選択]")
    results = {}

    for k in config.regime_candidates:
        print(f"    k={k} 推定中...", end="", flush=True)
        try:
            res = fit_hmm_robust(series, k, config)
            results[k] = res
            print(f" AIC={res.aic:.1f}, BIC={res.bic:.1f}")
        except Exception as e:
            print(f" 失敗: {e}")

    if results:
        best_aic = min(results, key=lambda k: results[k].aic)
        best_bic = min(results, key=lambda k: results[k].bic)
        print(f"    → AIC最小: {best_aic}, BIC最小: {best_bic}")

    return results


# ============================================================================
# 5. レジーム特性分析
# ============================================================================

@dataclass
class RegimeCharacteristics:
    posterior: pd.DataFrame
    viterbi: pd.Series
    regime_stats: pd.DataFrame
    regime_moments: pd.DataFrame
    transition_matrix: pd.DataFrame
    model_transition_matrix: pd.DataFrame
    expected_durations: dict[str, float]
    stationary_distribution: dict[str, float]
    regime_mapping: dict[int, int] | None = None


def compute_stationary_distribution(trans_mat: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(trans_mat.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    stationary = np.abs(np.real(eigvecs[:, idx]))
    return stationary / stationary.sum()


def weighted_stats(values: pd.Series, weights: pd.Series):
    mask = values.notna() & weights.notna()
    v, w = values[mask].values, weights[mask].values.astype(float)
    if len(v) == 0 or w.sum() == 0:
        return np.nan, np.nan
    mean = np.average(v, weights=w)
    var = np.average((v - mean) ** 2, weights=w)
    return mean, np.sqrt(var)


def analyze_regimes(hmm_result: HMMResult, df_factors: pd.DataFrame,
                    factor_col: str, config: Config) -> RegimeCharacteristics:
    posterior = hmm_result.posterior
    viterbi = hmm_result.viterbi
    model = hmm_result.model
    n_states = hmm_result.n_states

    factor = df_factors[factor_col].loc[posterior.index]
    ret = df_factors["Return"].loc[posterior.index]
    vol_raw = df_factors["VolatilityRaw"].loc[posterior.index]
    skew = df_factors["Skewness"].loc[posterior.index]
    kurt = df_factors["ExcessKurtosis"].loc[posterior.index]
    n_obs = len(posterior)

    stats_list = []
    moments_list = []

    for s in range(n_states):
        w = posterior[s]
        w_sum = w.sum()
        mean_factor, std_factor = weighted_stats(factor, w)
        mean_ret, _ = weighted_stats(ret, w)
        mean_vol, _ = weighted_stats(vol_raw, w)
        mean_skew, _ = weighted_stats(skew, w)
        mean_kurt, _ = weighted_stats(kurt, w)

        regime_mask = viterbi == s
        f_regime = factor[regime_mask]
        model_var = model.covars_[s, 0, 0] if model.covariance_type == "full" else model.covars_[s, 0]

        stats_list.append({
            "Regime": s, "Regime_Name": config.regime_names.get(s),
            "N_weight": w_sum, "Pct_weight": w_sum / n_obs * 100,
            "Factor_Mean": mean_factor, "Factor_Std": std_factor,
            "Return_Mean": mean_ret, "VolatilityRaw_Mean": mean_vol,
            "Skewness_Mean": mean_skew, "ExcessKurtosis_Mean": mean_kurt,
            "Model_Mean": model.means_[s, 0], "Model_Var": model_var,
        })
        moments_list.append({
            "Regime": s, "Regime_Name": config.regime_names.get(s),
            "Skew_Mean": mean_skew, "Kurt_Mean": mean_kurt,
        })

    # 遷移確率
    model_trans = model.transmat_
    trans_df = pd.DataFrame(
        model_trans,
        index=[config.regime_names.get(i) for i in range(n_states)],
        columns=[config.regime_names.get(i) for i in range(n_states)],
    )

    # 経験的遷移確率
    trans_counts = np.zeros((n_states, n_states))
    v = viterbi.values
    for i in range(1, len(v)):
        trans_counts[v[i-1], v[i]] += 1
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    empirical_trans = trans_counts / row_sums

    # 期待持続期間・定常分布
    expected_durations = {
        config.regime_names.get(i): 1 / (1 - model_trans[i, i]) if model_trans[i, i] < 1 else np.inf
        for i in range(n_states)
    }
    stationary_vec = compute_stationary_distribution(model_trans)
    stationary_distribution = {
        config.regime_names.get(i): stationary_vec[i] for i in range(n_states)
    }

    return RegimeCharacteristics(
        posterior=posterior, viterbi=viterbi,
        regime_stats=pd.DataFrame(stats_list),
        regime_moments=pd.DataFrame(moments_list),
        transition_matrix=pd.DataFrame(empirical_trans,
            index=[config.regime_names.get(i) for i in range(n_states)],
            columns=[config.regime_names.get(i) for i in range(n_states)]),
        model_transition_matrix=trans_df,
        expected_durations=expected_durations,
        stationary_distribution=stationary_distribution,
    )


def reorder_regimes(hmm_result: HMMResult, df_factors: pd.DataFrame,
                    factor_col: str, factor_type: str, config: Config):
    model = hmm_result.model
    n_states = hmm_result.n_states
    means = model.means_.flatten()

    if factor_type == "ascending":
        sorted_indices = np.argsort(means)
    else:
        sorted_indices = np.argsort(means)[::-1]

    mapping = {old: new for new, old in enumerate(sorted_indices)}

    print(f"  レジーム並べ替え ({factor_type}):")
    for old in sorted_indices:
        print(f"    Regime_{old} (mean={means[old]:.4f}) → {config.regime_names[mapping[old]]}")

    posterior_new = hmm_result.posterior.iloc[:, sorted_indices].copy()
    posterior_new.columns = list(range(n_states))
    viterbi_new = hmm_result.viterbi.map(mapping)

    new_model = copy.deepcopy(model)
    new_model.startprob_ = model.startprob_[sorted_indices]
    new_model.transmat_ = model.transmat_[sorted_indices][:, sorted_indices]
    new_model.means_ = model.means_[sorted_indices]
    new_model.covars_ = model.covars_[sorted_indices]

    hmm_new = HMMResult(
        model=new_model, n_states=n_states,
        log_likelihood=hmm_result.log_likelihood,
        aic=hmm_result.aic, bic=hmm_result.bic,
        posterior=posterior_new, viterbi=viterbi_new,
        converged=hmm_result.converged, n_iter_used=hmm_result.n_iter_used,
    )

    chars = analyze_regimes(hmm_new, df_factors, factor_col, config)
    chars.regime_mapping = mapping
    return hmm_new, chars


# ============================================================================
# 6. 平均イベント応答（相対日数プロット）
# ============================================================================

def compute_avg_event_response_all(factor_models: dict, window: int = 60):
    """全モデルの平均イベント応答を計算"""
    results = {}

    for model_key, (_, chars) in factor_models.items():
        crisis_prob = chars.posterior[2]
        dates = crisis_prob.index
        paths = []

        for ev in EVENTS:
            dt = pd.Timestamp(ev["date"])
            if dt < dates[0] or dt > dates[-1]:
                continue
            pos = dates.get_indexer([dt], method="nearest")[0]
            if pos - window < 0 or pos + window >= len(dates):
                continue
            seg = crisis_prob.iloc[pos - window:pos + window + 1].values
            if len(seg) == 2 * window + 1:
                paths.append(seg)

        if paths:
            arr = np.vstack(paths)
            mean_path = arr.mean(axis=0)
        else:
            mean_path = np.zeros(2 * window + 1)

        rel_days = np.arange(-window, window + 1)
        results[model_key] = (rel_days, mean_path, len(paths))

    return results


def plot_avg_event_response_all_models(factor_models: dict, config: Config,
                                        output_dir: Path, window: int = 60):
    """全4モデルの平均イベント応答プロット（視認性改善版）"""
    print("\n[平均イベント応答プロット]")

    responses = compute_avg_event_response_all(factor_models, window)

    fig, ax = plt.subplots(figsize=(12, 6))

    for model_key in ["logvol", "skew", "kurt", "skfactor"]:
        if model_key not in responses:
            continue
        rel_days, mean_path, n_events = responses[model_key]
        info = MODEL_INFO[model_key]
        ax.plot(rel_days, mean_path,
                label=f"{info['label']} (n={n_events})",
                color=info["color"], linestyle=info["linestyle"], lw=2)

    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5)

    ax.set_xlabel("イベント日からの営業日数", fontsize=12)
    ax.set_ylabel("平均 Crisis 確率", fontsize=12)
    ax.set_title("主要イベント前後の平均 Crisis 確率パス", fontsize=14, fontweight="bold")
    ax.set_xlim(-window, window)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "31_avg_event_response_all_models.pdf"
    plt.savefig(fig_path)
    print(f"  ✅ 保存: {fig_path}")
    plt.show()
    plt.close()


# ============================================================================
# 7. 個別イベントの詳細分析（全イベント）- 視認性改善版
# ============================================================================

def plot_all_events_crisis_probs(factor_models: dict, config: Config,
                                  output_dir: Path, window: int = 60):
    """全イベントの前後Crisis確率プロット（視認性改善版）"""
    print("\n[全イベントの詳細プロット]")

    base_key = "logvol" if "logvol" in factor_models else list(factor_models.keys())[0]
    _, base_chars = factor_models[base_key]
    base_dates = base_chars.posterior.index

    for ev in EVENTS:
        event_name = ev["name"]
        dt = pd.Timestamp(ev["date"])

        if dt < base_dates[0] or dt > base_dates[-1]:
            print(f"  - {event_name}: 対象期間外")
            continue

        pos = base_dates.get_indexer([dt], method="nearest")[0]
        if pos - window < 0 or pos + window >= len(base_dates):
            print(f"  - {event_name}: 端点に近い")
            continue

        window_idx = base_dates[pos - window: pos + window + 1]
        rel_days = np.arange(-window, window + 1)

        fig, ax = plt.subplots(figsize=(10, 5))

        for model_key in ["logvol", "skew", "kurt", "skfactor"]:
            if model_key not in factor_models:
                continue
            _, chars = factor_models[model_key]
            p = chars.posterior[2]
            idx = window_idx.intersection(p.index)
            if len(idx) == 0:
                continue
            p_values = p.loc[idx].values
            info = MODEL_INFO[model_key]
            ax.plot(rel_days[:len(p_values)], p_values,
                    label=info["label"], color=info["color"],
                    linestyle=info["linestyle"], lw=1.5)

        ax.axvline(0, color="black", lw=1.2, ls="--")
        ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.7)

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-window, window)
        ax.set_ylabel("Crisis 確率")
        ax.set_xlabel("イベント日からの営業日数")

        region_str = "（日本）" if ev.get("region") == "japan" else ""
        ax.set_title(f"{event_name}{region_str} ({ev['date']}) 前後のCrisis確率", fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

        plt.tight_layout()
        safe_name = event_name.replace("/", "_").replace(" ", "_")
        fig_path = output_dir / f"32_event_{dt.strftime('%Y%m%d')}_{safe_name}.pdf"
        plt.savefig(fig_path)
        print(f"  ✅ {event_name}")
        plt.show()
        plt.close()


def plot_selected_events_grid(factor_models: dict, config: Config, output_dir: Path, window: int = 60):
    """主要4イベントのグリッド表示（視認性改善版）"""
    selected = ["リーマンショック", "コロナショック", "チャイナショック", "令和のブラックマンデー"]
    event_dict = {ev["name"]: ev for ev in EVENTS}

    base_key = "logvol" if "logvol" in factor_models else list(factor_models.keys())[0]
    _, base_chars = factor_models[base_key]
    base_dates = base_chars.posterior.index

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, name in enumerate(selected):
        if name not in event_dict:
            continue
        ev = event_dict[name]
        dt = pd.Timestamp(ev["date"])
        ax = axes[i]

        if dt < base_dates[0] or dt > base_dates[-1]:
            continue
        pos = base_dates.get_indexer([dt], method="nearest")[0]
        if pos - window < 0 or pos + window >= len(base_dates):
            continue

        window_idx = base_dates[pos - window: pos + window + 1]
        rel_days = np.arange(-window, window + 1)

        for model_key in ["logvol", "skew", "kurt", "skfactor"]:
            if model_key not in factor_models:
                continue
            _, chars = factor_models[model_key]
            p = chars.posterior[2]
            idx = window_idx.intersection(p.index)
            if len(idx) == 0:
                continue
            p_values = p.loc[idx].values
            info = MODEL_INFO[model_key]
            ax.plot(rel_days[:len(p_values)], p_values,
                    label=info["label"] if i == 0 else "",
                    color=info["color"], linestyle=info["linestyle"], lw=1.3)

        ax.axvline(0, color="black", lw=1, ls="--")
        ax.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-window, window)
        ax.set_title(f"{name} ({ev['date']})", fontweight="bold")
        ax.set_xlabel("相対日数")
        ax.set_ylabel("Crisis確率")
        ax.grid(True, alpha=0.3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    axes[0].legend(loc="upper right", fontsize=9)
    plt.suptitle("主要イベント前後のCrisis確率比較", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig_path = output_dir / "33_selected_events_grid.pdf"
    plt.savefig(fig_path)
    print(f"  ✅ 保存: {fig_path}")
    plt.show()
    plt.close()


# ============================================================================
# 8. イベント前後での各因子（Vol, Skew, Kurt, SK）の推移
# ============================================================================

def plot_event_factor_dynamics(df_factors: pd.DataFrame, config: Config,
                                output_dir: Path, window: int = 60):
    """全イベント前後での各因子の推移"""
    print("\n[イベント前後の因子推移プロット]")

    factor_cols = {
        "LogVolatility": {"label": "log(Volatility)", "color": "#0072B2"},
        "Skewness": {"label": "Skewness", "color": "#009E73"},
        "ExcessKurtosis": {"label": "Excess Kurtosis", "color": "#CC79A7"},
        "SKFactor": {"label": "SK Factor", "color": "#D55E00"},
    }

    dates = df_factors.index

    for ev in EVENTS:
        event_name = ev["name"]
        dt = pd.Timestamp(ev["date"])

        if dt < dates[0] or dt > dates[-1]:
            continue

        pos = dates.get_indexer([dt], method="nearest")[0]
        if pos - window < 0 or pos + window >= len(dates):
            continue

        window_idx = dates[pos - window: pos + window + 1]
        rel_days = np.arange(-window, window + 1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, (col, info) in enumerate(factor_cols.items()):
            ax = axes[i]
            values = df_factors[col].loc[window_idx].values

            ax.plot(rel_days[:len(values)], values,
                    color=info["color"], lw=1.5, label=info["label"])
            ax.axvline(0, color="black", lw=1.2, ls="--", label="イベント日")
            ax.axhline(0, color="gray", lw=0.5, alpha=0.5)

            ax.set_xlabel("イベント日からの営業日数")
            ax.set_ylabel(info["label"])
            ax.set_title(f"{info['label']}", fontweight="bold")
            ax.set_xlim(-window, window)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        region_str = "（日本）" if ev.get("region") == "japan" else ""
        plt.suptitle(f"{event_name}{region_str} ({ev['date']}) 前後の因子推移",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        safe_name = event_name.replace("/", "_").replace(" ", "_")
        fig_path = output_dir / f"40_factors_{dt.strftime('%Y%m%d')}_{safe_name}.pdf"
        plt.savefig(fig_path)
        print(f"  ✅ {event_name}")
        plt.show()
        plt.close()


def plot_selected_events_factor_grid(df_factors: pd.DataFrame, config: Config,
                                      output_dir: Path, window: int = 60):
    """主要4イベントの因子推移グリッド表示"""
    print("\n[主要イベントの因子推移グリッド]")

    selected = ["リーマンショック", "コロナショック", "チャイナショック", "令和のブラックマンデー"]
    event_dict = {ev["name"]: ev for ev in EVENTS}

    factor_info = [
        ("LogVolatility", "log(Vol)", "#0072B2"),
        ("Skewness", "Skew", "#009E73"),
        ("ExcessKurtosis", "Ex.Kurt", "#CC79A7"),
        ("SKFactor", "SK Factor", "#D55E00"),
    ]

    dates = df_factors.index

    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    for col_idx, name in enumerate(selected):
        if name not in event_dict:
            continue
        ev = event_dict[name]
        dt = pd.Timestamp(ev["date"])

        if dt < dates[0] or dt > dates[-1]:
            continue
        pos = dates.get_indexer([dt], method="nearest")[0]
        if pos - window < 0 or pos + window >= len(dates):
            continue

        window_idx = dates[pos - window: pos + window + 1]
        rel_days = np.arange(-window, window + 1)

        for row_idx, (col, label, color) in enumerate(factor_info):
            ax = axes[row_idx, col_idx]
            values = df_factors[col].loc[window_idx].values

            ax.plot(rel_days[:len(values)], values, color=color, lw=1.2)
            ax.axvline(0, color="black", lw=1, ls="--")
            ax.axhline(0, color="gray", lw=0.3, alpha=0.5)

            if row_idx == 0:
                ax.set_title(f"{name}\n({ev['date']})", fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10)
            if row_idx == 3:
                ax.set_xlabel("相対日数", fontsize=9)

            ax.set_xlim(-window, window)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    plt.suptitle("主要イベント前後の各因子推移", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig_path = output_dir / "41_selected_events_factors_grid.pdf"
    plt.savefig(fig_path)
    print(f"  ✅ 保存: {fig_path}")
    plt.show()
    plt.close()


def plot_avg_factor_response(df_factors: pd.DataFrame, config: Config,
                              output_dir: Path, window: int = 60):
    """全イベントの平均因子推移"""
    print("\n[平均因子推移プロット]")

    factor_cols = {
        "LogVolatility": {"label": "log(Volatility)", "color": "#0072B2"},
        "Skewness": {"label": "Skewness", "color": "#009E73"},
        "ExcessKurtosis": {"label": "Excess Kurtosis", "color": "#CC79A7"},
        "SKFactor": {"label": "SK Factor", "color": "#D55E00"},
    }

    dates = df_factors.index
    rel_days = np.arange(-window, window + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (col, info) in enumerate(factor_cols.items()):
        ax = axes[i]
        paths = []

        for ev in EVENTS:
            dt = pd.Timestamp(ev["date"])
            if dt < dates[0] or dt > dates[-1]:
                continue
            pos = dates.get_indexer([dt], method="nearest")[0]
            if pos - window < 0 or pos + window >= len(dates):
                continue
            seg = df_factors[col].iloc[pos - window:pos + window + 1].values
            if len(seg) == 2 * window + 1:
                paths.append(seg)

        if paths:
            arr = np.vstack(paths)
            mean_path = arr.mean(axis=0)
            std_path = arr.std(axis=0)

            ax.plot(rel_days, mean_path, color=info["color"], lw=2,
                    label=f"平均 (n={len(paths)})")
            ax.fill_between(rel_days, mean_path - std_path, mean_path + std_path,
                           color=info["color"], alpha=0.2, label="±1σ")

        ax.axvline(0, color="black", lw=1.2, ls="--")
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)

        ax.set_xlabel("イベント日からの営業日数")
        ax.set_ylabel(info["label"])
        ax.set_title(f"{info['label']}", fontweight="bold")
        ax.set_xlim(-window, window)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("主要イベント前後の平均因子推移 (全イベント)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig_path = output_dir / "42_avg_factor_response.pdf"
    plt.savefig(fig_path)
    print(f"  ✅ 保存: {fig_path}")
    plt.show()
    plt.close()


# ============================================================================
# 9. 基本可視化
# ============================================================================

def plot_regime_probabilities_compact(chars: RegimeCharacteristics, df_factors: pd.DataFrame,
                                       factor_col: str, factor_label: str,
                                       config: Config, save_path: str):
    """レジーム確率の時系列プロット"""
    posterior = chars.posterior
    viterbi = chars.viterbi
    factor = df_factors[factor_col].loc[posterior.index]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1.2, 1]})

    # (a) 因子 + レジーム背景
    ax = axes[0]
    ax.plot(factor.index, factor.values, lw=0.6, color="#333333")
    ax.axhline(0, color="black", lw=0.5)

    for i in range(len(factor) - 1):
        s = int(viterbi.iloc[i])
        ax.axvspan(factor.index[i], factor.index[i + 1],
                   alpha=0.2, color=config.regime_colors.get(s, "gray"), lw=0)

    ax.set_ylabel(factor_label)
    ax.set_title(f"(a) {factor_label} の時系列とレジーム", fontweight="bold")
    add_event_lines(ax, show_labels=True)

    legend_elems = [Patch(facecolor=config.regime_colors[i], alpha=0.5,
                          label=config.regime_names[i]) for i in range(3)]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=9)

    # (b) Crisis確率
    ax = axes[1]
    ax.fill_between(posterior.index, 0, posterior[2],
                    color=config.regime_colors[2], alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Crisis 確率")
    ax.set_xlabel("日付")
    ax.set_title("(b) Crisis レジーム確率", fontweight="bold")
    add_event_lines(ax, show_labels=False)

    plt.suptitle(f"Gaussian HMM 結果 ({factor_label})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  ✅ 保存: {save_path}")
    plt.show()
    plt.close()


# ============================================================================
# 10. ★改善版★ レジーム特性サマリープロット
# ============================================================================

def plot_regime_characteristics_improved(chars: RegimeCharacteristics, factor_label: str,
                                          config: Config, save_path: str):
    """
    レジーム特性サマリー（改善版）

    改善点:
    - 観測割合と定常分布を並べて比較
    - 期待持続期間を独立したサブプロットに
    - 文字の重なりを解消
    - 2x2のグリッドレイアウト
    """
    stats = chars.regime_stats
    trans = chars.model_transition_matrix
    durations = chars.expected_durations
    stationary = chars.stationary_distribution

    n_regimes = len(stats)
    regime_names = stats["Regime_Name"].tolist()
    colors = [config.regime_colors.get(i, "gray") for i in range(n_regimes)]

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, wspace=0.3, hspace=0.35)

    # =========================================================================
    # (a) 観測割合 vs 定常分布の比較
    # =========================================================================
    ax = fig.add_subplot(gs[0, 0])

    x = np.arange(n_regimes)
    width = 0.35

    # 観測割合（経験的分布）
    obs_pct = stats["Pct_weight"].values
    bars1 = ax.bar(x - width/2, obs_pct, width, color=colors, alpha=0.8,
                   label="観測割合（経験的）", edgecolor="black", linewidth=0.5)

    # 定常分布（理論的）
    stat_pct = [stationary[name] * 100 for name in regime_names]
    bars2 = ax.bar(x + width/2, stat_pct, width, color=colors, alpha=0.4,
                   hatch="//", label="定常分布（理論的）", edgecolor="black", linewidth=0.5)

    # バーの上に数値を表示
    for bar, val in zip(bars1, obs_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    for bar, val in zip(bars2, stat_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, style="italic")

    ax.set_ylabel("割合 (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(regime_names, fontsize=11)
    ax.set_title("(a) レジーム分布: 観測割合 vs 定常分布", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(max(obs_pct), max(stat_pct)) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    # =========================================================================
    # (b) 期待持続期間
    # =========================================================================
    ax = fig.add_subplot(gs[0, 1])

    dur_vals = [min(durations[name], 200) for name in regime_names]  # 上限200日
    bars = ax.bar(x, dur_vals, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)

    # バーの上に数値を表示
    for bar, name in zip(bars, regime_names):
        val = durations[name]
        if val > 200:
            label = f"{val:.0f}日"
        else:
            label = f"{val:.1f}日"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                label, ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("期待持続日数", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(regime_names, fontsize=11)
    ax.set_title("(b) 期待持続期間（1/(1-p_ii)）", fontweight="bold", fontsize=12)
    ax.set_ylim(0, max(dur_vals) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    # 計算式の注釈
    ax.text(0.98, 0.02, r"$E[D_i] = \frac{1}{1 - p_{ii}}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, style="italic", color="gray")

    # =========================================================================
    # (c) 遷移確率行列
    # =========================================================================
    ax = fig.add_subplot(gs[1, 0])

    im = ax.imshow(trans.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(trans.columns)))
    ax.set_yticks(range(len(trans.index)))
    ax.set_xticklabels(trans.columns, fontsize=11)
    ax.set_yticklabels(trans.index, fontsize=11)
    ax.set_xlabel("To（遷移先）", fontsize=11)
    ax.set_ylabel("From（遷移元）", fontsize=11)
    ax.set_title("(c) 遷移確率行列", fontweight="bold", fontsize=12)

    # 各セルに確率を表示
    for i in range(len(trans.index)):
        for j in range(len(trans.columns)):
            v = trans.iloc[i, j]
            text_color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=text_color, fontsize=12, fontweight="bold" if i == j else "normal")

    # カラーバー
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("遷移確率", fontsize=10)

    # =========================================================================
    # (d) レジーム別条件付きモーメント（ボラティリティ追加）
    # =========================================================================
    ax = fig.add_subplot(gs[1, 1])

    # テーブル形式で表示
    ax.axis("off")

    # データ準備（ボラティリティ追加）
    table_data = []
    headers = ["レジーム", "Vol (年率)", "Return", "Skewness", "Ex.Kurt"]

    for _, row in stats.iterrows():
        # ボラティリティは年率換算（日次×√252）で表示
        vol_annualized = row['VolatilityRaw_Mean'] * np.sqrt(252) * 100
        table_data.append([
            row["Regime_Name"],
            f"{vol_annualized:.1f}%",
            f"{row['Return_Mean']*100:.3f}%",
            f"{row['Skewness_Mean']:.3f}",
            f"{row['ExcessKurtosis_Mean']:.2f}"
        ])

    # テーブル作成
    table = ax.table(cellText=table_data, colLabels=headers,
                     loc="center", cellLoc="center",
                     colColours=["#E8E8E8"]*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.8)

    # ヘッダー行のスタイル
    for j in range(len(headers)):
        table[(0, j)].set_text_props(fontweight="bold")

    # レジーム列に色を付ける
    for i, (idx, row) in enumerate(stats.iterrows()):
        table[(i+1, 0)].set_facecolor(colors[i])
        table[(i+1, 0)].set_alpha(0.3)

    # 注釈
    ax.text(0.5, -0.02, "※ Vol = 日次σ × √252 で年率換算",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, style="italic", color="gray")

    ax.set_title("(d) レジーム別条件付きモーメント", fontweight="bold", fontsize=12, pad=20)

    # =========================================================================
    # 全体タイトル
    # =========================================================================
    plt.suptitle(f"レジーム特性サマリー ({factor_label})", fontsize=15, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    print(f"  ✅ 保存: {save_path}")
    plt.show()
    plt.close()


def plot_regime_characteristics_compact(chars: RegimeCharacteristics, factor_label: str,
                                         config: Config, save_path: str):
    """
    レジーム特性サマリー（旧版 - 互換性のため残す）
    新しい改善版は plot_regime_characteristics_improved を使用
    """
    # 改善版を呼び出す
    plot_regime_characteristics_improved(chars, factor_label, config, save_path)


# ============================================================================
# 11. モデル間比較（時系列カラーマップ）
# ============================================================================

def plot_regime_agreement_timeline(factor_models: dict, config: Config, output_dir: Path):
    """モデル間のレジーム一致を時系列カラーマップで表示"""
    print("\n[モデル間レジーム一致度（時系列）]")

    model_keys = [k for k in ["logvol", "skew", "kurt", "skfactor"] if k in factor_models]
    if len(model_keys) < 2:
        return

    # Viterbi系列取得
    viterbi_dict = {k: factor_models[k][1].viterbi for k in model_keys}
    common_idx = viterbi_dict[model_keys[0]].index
    for k in model_keys[1:]:
        common_idx = common_idx.intersection(viterbi_dict[k].index)

    # カテゴリ計算
    n = len(common_idx)
    category = np.zeros(n, dtype=int)

    for i, dt in enumerate(common_idx):
        states = [viterbi_dict[k].loc[dt] for k in model_keys]
        if all(s == 0 for s in states):
            category[i] = 0
        elif all(s == 1 for s in states):
            category[i] = 1
        elif all(s == 2 for s in states):
            category[i] = 2
        else:
            category[i] = 3

    # 集計
    unique, counts = np.unique(category, return_counts=True)
    print("  カテゴリ別日数:")
    labels = {0: "全Normal", 1: "全Stress", 2: "全Crisis", 3: "不一致"}
    for u, c in zip(unique, counts):
        print(f"    {labels[u]}: {c}日 ({c/n*100:.1f}%)")

    # プロット
    fig, ax = plt.subplots(figsize=(16, 3))

    cmap_colors = ["#009E73", "#E69F00", "#D55E00", "#999999"]
    cmap = ListedColormap(cmap_colors)

    y = np.zeros(n)
    scatter = ax.scatter(common_idx, y, c=category, cmap=cmap, s=8, marker="s", vmin=0, vmax=3)

    ax.set_yticks([])
    ax.set_xlabel("日付")
    ax.set_title("4モデルのレジーム一致状況（緑=全Normal, 橙=全Stress, 赤=全Crisis, 灰=不一致）",
                 fontweight="bold")
    add_event_lines(ax, show_labels=True, label_fontsize=6)

    legend_elems = [Patch(facecolor=cmap_colors[i], label=labels[i]) for i in range(4)]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)

    plt.tight_layout()
    fig_path = output_dir / "34_regime_agreement_timeline.pdf"
    plt.savefig(fig_path)
    print(f"  ✅ 保存: {fig_path}")
    plt.show()
    plt.close()


def plot_crisis_prob_comparison(factor_models: dict, config: Config, output_dir: Path):
    """全4モデルのCrisis確率時系列比較"""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    for i, model_key in enumerate(["logvol", "skew", "kurt", "skfactor"]):
        if model_key not in factor_models:
            continue
        ax = axes[i]
        _, chars = factor_models[model_key]
        p = chars.posterior[2]
        info = MODEL_INFO[model_key]

        ax.fill_between(p.index, 0, p, color=info["color"], alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Crisis 確率")
        ax.set_title(f"{info['label']} モデル", fontweight="bold", fontsize=11)
        add_event_lines(ax, show_labels=(i == 0), label_fontsize=6)

    axes[-1].set_xlabel("日付")
    plt.suptitle("Crisis確率の時系列比較（全4モデル）", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig_path = output_dir / "35_crisis_prob_comparison_4models.pdf"
    plt.savefig(fig_path)
    print(f"  ✅ 保存: {fig_path}")
    plt.show()
    plt.close()


# ============================================================================
# 12. 追加分析（論文に必要）
# ============================================================================

def plot_conditional_return_distribution(factor_models: dict, df_factors: pd.DataFrame,
                                          config: Config, output_dir: Path):
    """レジーム別のリターン分布（各モデル）"""
    print("\n[条件付きリターン分布]")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, model_key in enumerate(["logvol", "skew", "kurt", "skfactor"]):
        if model_key not in factor_models:
            continue
        ax = axes[i]
        _, chars = factor_models[model_key]
        viterbi = chars.viterbi
        ret = df_factors["Return"].loc[viterbi.index] * 100

        for s in range(3):
            data = ret[viterbi == s].dropna()
            if len(data) > 10:
                ax.hist(data, bins=50, density=True, alpha=0.5,
                        color=config.regime_colors[s],
                        label=f"{config.regime_names[s]} (n={len(data)})")

        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("日次リターン (%)")
        ax.set_ylabel("密度")
        ax.set_title(f"{MODEL_INFO[model_key]['label']} モデル", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(-10, 10)

    plt.suptitle("レジーム別リターン分布", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig_path = output_dir / "36_conditional_return_distribution.pdf"
    plt.savefig(fig_path)
    print(f"  ✅ 保存: {fig_path}")
    plt.show()
    plt.close()


def plot_moment_space_regimes(factor_models: dict, df_factors: pd.DataFrame,
                               config: Config, output_dir: Path):
    """Skew vs Kurt のモーメント空間でのレジーム可視化"""
    print("\n[モーメント空間でのレジーム可視化]")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    skew = df_factors["Skewness"]
    kurt = df_factors["ExcessKurtosis"]

    for i, model_key in enumerate(["logvol", "skew", "kurt", "skfactor"]):
        if model_key not in factor_models:
            continue
        ax = axes[i]
        _, chars = factor_models[model_key]
        viterbi = chars.viterbi

        common_idx = skew.index.intersection(viterbi.index)

        for s in range(3):
            mask = viterbi.loc[common_idx] == s
            ax.scatter(skew.loc[common_idx][mask], kurt.loc[common_idx][mask],
                       s=3, alpha=0.3, color=config.regime_colors[s],
                       label=config.regime_names[s])

        ax.axhline(0, color="black", lw=0.5, alpha=0.5)
        ax.axvline(0, color="black", lw=0.5, alpha=0.5)
        ax.set_xlabel("Skewness")
        ax.set_ylabel("Excess Kurtosis")
        ax.set_title(f"{MODEL_INFO[model_key]['label']} モデル", fontweight="bold")
        ax.legend(fontsize=8, markerscale=3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 15)

    plt.suptitle("モーメント空間でのレジーム分布 (Skew vs Kurt)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig_path = output_dir / "37_moment_space_regimes.pdf"
    plt.savefig(fig_path)
    print(f"  ✅ 保存: {fig_path}")
    plt.show()
    plt.close()


def plot_cumulative_crisis_days(factor_models: dict, config: Config, output_dir: Path):
    """累積Crisis日数の時系列"""
    print("\n[累積Crisis日数]")

    fig, ax = plt.subplots(figsize=(14, 6))

    for model_key in ["logvol", "skew", "kurt", "skfactor"]:
        if model_key not in factor_models:
            continue
        _, chars = factor_models[model_key]
        viterbi = chars.viterbi
        is_crisis = (viterbi == 2).astype(int)
        cum_crisis = is_crisis.cumsum()

        info = MODEL_INFO[model_key]
        ax.plot(cum_crisis.index, cum_crisis.values,
                label=info["label"], color=info["color"],
                linestyle=info["linestyle"], lw=1.5)

    ax.set_xlabel("日付")
    ax.set_ylabel("累積Crisis日数")
    ax.set_title("累積Crisis日数の推移（各モデル比較）", fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    add_event_lines(ax, show_labels=True, label_fontsize=6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "38_cumulative_crisis_days.pdf"
    plt.savefig(fig_path)
    print(f"  ✅ 保存: {fig_path}")
    plt.show()
    plt.close()


def generate_latex_summary_table(factor_models: dict, config: Config, output_dir: Path):
    """LaTeX用のサマリー表を生成"""
    print("\n[LaTeX用サマリー表]")

    for model_key in ["logvol", "skew", "kurt", "skfactor"]:
        if model_key not in factor_models:
            continue
        _, chars = factor_models[model_key]
        stats = chars.regime_stats[["Regime_Name", "Pct_weight", "Return_Mean",
                                     "Skewness_Mean", "ExcessKurtosis_Mean"]]
        stats.columns = ["Regime", "Weight(%)", "Return", "Skewness", "Ex.Kurt"]

        latex = stats.to_latex(index=False, float_format="%.3f",
                               caption=f"{MODEL_INFO[model_key]['label']}モデルのレジーム特性",
                               label=f"tab:regime_characteristics_{model_key}")

        with open(output_dir / f"table_regime_characteristics_{model_key}.tex", "w") as f:
            f.write(latex)
        print(f"  ✅ table_regime_characteristics_{model_key}.tex 保存")


# ============================================================================
# 13. CSV出力
# ============================================================================

def save_regime_probabilities(chars: RegimeCharacteristics, df_factors: pd.DataFrame,
                              config: Config, suffix: str, output_dir: Path):
    posterior = chars.posterior.copy()
    posterior.columns = ["P_Normal", "P_Stress", "P_Crisis"]
    out = posterior.copy()
    out["Regime"] = chars.viterbi
    out["Regime_Name"] = out["Regime"].map(config.regime_names)
    out["Return"] = df_factors["Return"]
    out["Skewness"] = df_factors["Skewness"]
    out["ExcessKurtosis"] = df_factors["ExcessKurtosis"]

    csv_path = output_dir / f"regime_probabilities_{suffix}.csv"
    out.to_csv(csv_path)
    print(f"  ✅ {csv_path.name}")


def save_characteristics(chars: RegimeCharacteristics, suffix: str, output_dir: Path):
    chars.regime_stats.to_csv(output_dir / f"regime_stats_{suffix}.csv", index=False)
    chars.model_transition_matrix.to_csv(output_dir / f"transition_matrix_{suffix}.csv")


# ============================================================================
# 14. メイン実行
# ============================================================================

def main():
    print("=" * 70)
    print("Step 2: 3レジーム Gaussian HMM 分析（完全統合版）- 可視化改善版")
    print("=" * 70)

    config = Config()
    setup_matplotlib()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"\n出力ディレクトリ: {output_dir.absolute()}")

    # 1. データ読み込み
    df = load_data(config)
    if df is None:
        return None

    # 2. 危機因子構築
    df_factors = construct_factors(df)

    # 3. 危機因子設定
    factor_settings = [
        {"key": "logvol", "factor_col": "LogVolatility",
         "display_name": "log(Volatility)", "suffix": "logvol", "factor_type": "ascending"},
        {"key": "skew", "factor_col": "Skewness",
         "display_name": "Skewness", "suffix": "skew", "factor_type": "descending"},
        {"key": "kurt", "factor_col": "ExcessKurtosis",
         "display_name": "Excess Kurtosis", "suffix": "kurt", "factor_type": "ascending"},
        {"key": "skfactor", "factor_col": "SKFactor",
         "display_name": "SK Factor", "suffix": "skfactor", "factor_type": "ascending"},
    ]

    # 4. HMM推定
    selection_results = {}
    for fs in factor_settings:
        print("\n" + "=" * 60)
        print(f"[{fs['display_name']}]")
        print("=" * 60)
        series = df_factors[fs["factor_col"]]
        selection_results[fs["key"]] = select_num_states(series, config)

    # 5. 3レジームモデルの詳細分析
    factor_models = {}

    for fs in factor_settings:
        key = fs["key"]
        sel = selection_results.get(key, {})
        if config.n_states not in sel:
            continue

        print("\n" + "-" * 50)
        print(f"[{fs['display_name']}] 3レジームモデル")
        print("-" * 50)

        hmm_raw = sel[config.n_states]
        hmm_reordered, chars = reorder_regimes(
            hmm_raw, df_factors, fs["factor_col"], fs["factor_type"], config
        )
        factor_models[key] = (hmm_reordered, chars)

        # 基本統計
        print("\n  レジーム特性:")
        print(chars.regime_stats[["Regime_Name", "Pct_weight", "Return_Mean"]].to_string(index=False))
        print(f"\n  期待持続期間: {chars.expected_durations}")
        print(f"  定常分布: {chars.stationary_distribution}")

        # 基本可視化
        plot_regime_probabilities_compact(
            chars, df_factors, fs["factor_col"], fs["display_name"], config,
            str(output_dir / f"10_regime_prob_{fs['suffix']}.pdf")
        )

        # ★改善版レジーム特性プロット★
        plot_regime_characteristics_improved(
            chars, fs["display_name"], config,
            str(output_dir / f"11_regime_chars_{fs['suffix']}.pdf")
        )

        # CSV出力
        save_regime_probabilities(chars, df_factors, config, fs["suffix"], output_dir)
        save_characteristics(chars, fs["suffix"], output_dir)

    # 6. 平均イベント応答
    print("\n" + "=" * 70)
    print("イベント応答分析")
    print("=" * 70)
    plot_avg_event_response_all_models(factor_models, config, output_dir)

    # 7. 全イベントの詳細プロット
    plot_all_events_crisis_probs(factor_models, config, output_dir)
    plot_selected_events_grid(factor_models, config, output_dir)

    # 8. イベント前後の因子推移
    print("\n" + "=" * 70)
    print("イベント前後の因子推移分析")
    print("=" * 70)
    plot_event_factor_dynamics(df_factors, config, output_dir)
    plot_selected_events_factor_grid(df_factors, config, output_dir)
    plot_avg_factor_response(df_factors, config, output_dir)

    # 9. モデル間比較
    print("\n" + "=" * 70)
    print("モデル間比較分析")
    print("=" * 70)
    plot_regime_agreement_timeline(factor_models, config, output_dir)
    plot_crisis_prob_comparison(factor_models, config, output_dir)

    # 10. 追加分析（論文に重要）
    print("\n" + "=" * 70)
    print("追加分析（論文用）")
    print("=" * 70)
    plot_conditional_return_distribution(factor_models, df_factors, config, output_dir)
    plot_moment_space_regimes(factor_models, df_factors, config, output_dir)
    plot_cumulative_crisis_days(factor_models, config, output_dir)
    generate_latex_summary_table(factor_models, config, output_dir)

    # 完了サマリー
    print("\n" + "=" * 70)
    print("Step 2 完了サマリー")
    print("=" * 70)
    print(f"  出力ディレクトリ: {output_dir.absolute()}")
    print("\n  主要な出力ファイル:")
    print("    [基本分析]")
    print("      10-11_regime_*: 各モデルのレジーム分析（可視化改善版）")
    print("    [イベント応答分析]")
    print("      31_avg_event_response_all_models.pdf")
    print("    [イベント詳細]")
    print("      32_event_*.pdf: 全イベントの詳細プロット")
    print("      33_selected_events_grid.pdf")
    print("    [イベント前後の因子推移]")
    print("      40_factors_*.pdf: 各イベントの因子推移")
    print("      41_selected_events_factors_grid.pdf")
    print("      42_avg_factor_response.pdf")
    print("    [モデル比較]")
    print("      34_regime_agreement_timeline.pdf")
    print("      35_crisis_prob_comparison_4models.pdf")
    print("    [追加分析]")
    print("      36_conditional_return_distribution.pdf")
    print("      37_moment_space_regimes.pdf")
    print("      38_cumulative_crisis_days.pdf")
    print("    [LaTeX用]")
    print("      table_*.tex")

    return {
        "df_factors": df_factors,
        "factor_models": factor_models,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    results = main()