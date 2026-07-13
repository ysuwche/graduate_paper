"""
卒業論文：リターン分布形状のレジームダイナミクス分析
第1段階（最終版）：日経225専用データ取得・加工スクリプト
============================================

このスクリプトの目的（卒論用）:
✔ 日経225データ（価格・リターン）の作成
✔ 欠損処理・基本統計量の計算
✔ 日経225のイベントタイムライン図
✔ リターン分布（ヒスト＋QQ図）
✔ 後続の EWMA・MSモデル・リスク評価に使うCSVの生成

出力ファイル:
- nikkei_prices.csv              : 価格データ
- nikkei_returns.csv             : リターンデータ
- nikkei_statistics.csv          : 基本統計量
- fig_price.pdf(png)             : 価格推移図（イベント付き）
- fig_return_distribution.pdf(png): リターン分布＋QQ図
- fig_event_timeline.pdf(png)    : リターン＋イベントタイムライン図
============================================
"""

# 依存ライブラリは requirements.txt からインストールしてください:
#   pip install -r requirements.txt
# (Jupyter/Colab で個別に入れる場合は下行のコメントを解除)
# !pip install japanize_matplotlib yfinance

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.stats import skew, kurtosis, jarque_bera, norm, probplot
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------
# 1. 主要設定
# ----------------------------------------------
START = "2000-01-01"
END = "2024-12-31"
TICKER = "^N225"  # 日経平均株価

# 卒論分析用イベントリスト（ユーザー指定の13イベント）
# 日付は早期検知検証のため、暴落の「起点」または「決定的なトリガー日」を設定

EVENTS = [
    # 1. ITバブル崩壊 (NASDAQのピークアウト日)
    {"name": "ITバブル崩壊", "date": "2000-03-10", "region": "global"},

    # 2. 9.11テロ (発生日)
    {"name": "9.11テロ", "date": "2001-09-11", "region": "global"},

    # 4. サブプライム (BNPパリバ・ショック：世界同時株安の起点)
    {"name": "サブプライム", "date": "2007-08-09", "region": "global"},

    # 5. リーマンショック (破綻申請日)
    {"name": "リーマンショック", "date": "2008-09-15", "region": "global"},

    # 6. 東日本大震災 (発生日)
    {"name": "東日本大震災", "date": "2011-03-11", "region": "japan"},

    # 7. バーナンキショック (5.23ショック：暴落開始日)
    {"name": "バーナンキショック", "date": "2013-05-23", "region": "global"},

    # 8. チャイナショック (人民元切り下げ：世界波及のトリガー日)
    {"name": "チャイナショック", "date": "2015-08-11", "region": "global"},

    # 9. Brexit (国民投票結果判明・暴落日)
    {"name": "Brexit", "date": "2016-06-24", "region": "global"},

    {"name": "トランプ当選", "date": "2016-11-09", "region": "global"},

    # 10. 2018年末株安 (米金利上昇懸念による下落トレンド入り起点)
    {"name": "クリスマスショック", "date": "2018-12-25", "region": "global"},

    # 11. コロナショック (世界同時株安の開始日)
    {"name": "コロナショック", "date": "2020-02-24", "region": "global"},

    # 12. ウクライナ侵攻 (侵攻開始日)
    {"name": "ウクライナ侵攻", "date": "2022-02-24", "region": "global"},

    # 13. 令和のブラックマンデー (歴史的大暴落当日)
    {"name": "令和のブラックマンデー", "date": "2024-08-05", "region": "japan"}
]


# ----------------------------------------------
# 2. データ取得
# ----------------------------------------------
def download_nikkei(use_sample: bool = False) -> pd.DataFrame:
    """
    日経225データをyfinanceから取得
    use_sample=True の場合はオフライン環境用の疑似データを生成
    """
    print("=" * 50)
    print("日経225データ取得中...")
    print("=" * 50)

    if use_sample:
        # オフライン環境用：簡易な擬似データ
        print("※ サンプルデータモード（デモ用）")
        np.random.seed(42)
        dates = pd.bdate_range(start=START, end=END)
        n = len(dates)

        mu = 0.02 / 252           # 年率2%程度
        sigma = 0.20 / np.sqrt(252)  # 年率ボラ20%程度

        returns = np.zeros(n)

        for i in range(n):
            date = dates[i]
            if (pd.Timestamp('2008-09-01') <= date <= pd.Timestamp('2009-03-31')) or \
               (pd.Timestamp('2020-02-20') <= date <= pd.Timestamp('2020-04-30')):
                # 危機期間（高ボラ・負の期待リターン）
                returns[i] = np.random.normal(-0.002, sigma * 2.5)
            elif pd.Timestamp('2013-01-01') <= date <= pd.Timestamp('2015-06-30'):
                # アベノミクス的な上昇局面
                returns[i] = np.random.normal(0.001, sigma * 1.2)
            else:
                # 通常期間
                returns[i] = np.random.normal(mu, sigma)

        prices = 18000 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({'Nikkei225': prices}, index=dates)
        df.index.name = 'Date'
    else:
        try:
            df = yf.download(TICKER, start=START, end=END, progress=False)

            if df.empty:
                raise ValueError("データ取得失敗")

            # yfinanceのMultiIndex対応
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[['Close']].copy()
            df.columns = ['Nikkei225']

            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'

        except Exception as e:
            print(f"⚠️ データ取得エラー: {e}")
            print("サンプルデータモードに切り替えます...")
            return download_nikkei(use_sample=True)

    print(f"取得期間: {df.index.min().strftime('%Y-%m-%d')} → {df.index.max().strftime('%Y-%m-%d')}")
    print(f"取得行数: {len(df):,} 営業日")

    return df


# ----------------------------------------------
# 3. リターン計算
# ----------------------------------------------
def calc_returns(df: pd.DataFrame) -> pd.DataFrame:
    """終値から日次対数リターンを計算"""
    df = df.copy()
    df['Return'] = np.log(df['Nikkei225'] / df['Nikkei225'].shift(1))
    df = df.dropna()

    print(f"\nリターン計算完了: {len(df):,} 観測値")
    print(f"  平均リターン（日次）: {df['Return'].mean() * 100:.4f}%")
    print(f"  標準偏差（日次）    : {df['Return'].std() * 100:.4f}%")

    return df


# ----------------------------------------------
# 4. 統計量計算
# ----------------------------------------------
def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """リターンの詳細統計量を計算し、DataFrameで返す"""
    r = df['Return']
    jb_stat, jb_p = jarque_bera(r)

    stats = {
        'Observations': len(r),
        'Mean': r.mean(),
        'Std': r.std(),
        'Min': r.min(),
        'Max': r.max(),
        'Skewness': skew(r),
        'Excess_Kurtosis': kurtosis(r, fisher=True),
        'JB_Statistic': jb_stat,
        'JB_pvalue': jb_p,
        'Annualized_Mean': r.mean() * 252,
        'Annualized_Std': r.std() * np.sqrt(252),
    }

    stats_df = pd.DataFrame([stats])

    print("\n" + "=" * 50)
    print("基本統計量")
    print("=" * 50)
    print(f"  観測数:           {stats['Observations']:,}")
    print(f"  平均（日次）:     {stats['Mean']*100:.4f}%")
    print(f"  標準偏差（日次）: {stats['Std']*100:.4f}%")
    print(f"  歪度:             {stats['Skewness']:.4f}")
    print(f"  超過尖度:         {stats['Excess_Kurtosis']:.4f}")
    print(f"  最小値:           {stats['Min']*100:.4f}%")
    print(f"  最大値:           {stats['Max']*100:.4f}%")
    print(f"  JB統計量:         {stats['JB_Statistic']:.2f} (p={stats['JB_pvalue']:.2e})")
    print(f"  年率リターン:     {stats['Annualized_Mean']*100:.2f}%")
    print(f"  年率ボラ:         {stats['Annualized_Std']*100:.2f}%")

    return stats_df


# ----------------------------------------------
# 5. 可視化：価格推移＋イベント
# ----------------------------------------------
def plot_price(df: pd.DataFrame,
               save_path_pdf: str = "fig_price.pdf",
               save_path_png: str | None = None) -> None:
    """
    日経225 価格推移 + 主要イベント（縦線＋イベント名）
    論文用：単独図／PDF出力
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # 価格時系列
    ax.plot(df.index, df['Nikkei225'],
            linewidth=0.8,
            label='日経225')

    y_min = df['Nikkei225'].min()
    y_max = df['Nikkei225'].max()

    # イベント描画
    for ev in EVENTS:
        ts = pd.Timestamp(ev["date"])
        if ts < df.index.min() or ts > df.index.max():
            continue

        color = "red" if ev["region"] == "global" else "blue"
        linestyle = "--" if ev["region"] == "global" else ":"

        ax.axvline(ts, color=color, ls=linestyle, alpha=0.8, linewidth=1.0)

        # global は上、japan は下にラベル
        if ev["region"] == "global":
            y_text = y_max * 0.98
            va = "top"
        else:
            y_text = y_min * 1.02
            va = "bottom"

        ax.text(ts, y_text, ev["name"],
                rotation=90,
                fontsize=7,
                color=color,
                ha="right",
                va=va)

    ax.set_title("日経平均株価の推移と主要イベント（2000–2024）",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("日付", fontsize=11)
    ax.set_ylabel("価格（円）", fontsize=11)
    ax.grid(alpha=0.3, linestyle="-")
    ax.set_xlim(df.index.min(), df.index.max())

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", linewidth=1, label="日経225"),
        Line2D([0], [0], color="red", linestyle="--", label="グローバルイベント"),
        Line2D([0], [0], color="blue", linestyle=":", label="日本のイベント"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()

    plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"保存: {save_path_pdf}")

    if save_path_png is not None:
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"保存: {save_path_png}")

    plt.show()


# ----------------------------------------------
# 6. 可視化：リターン分布＋QQ
# ----------------------------------------------
def plot_return_distribution(df: pd.DataFrame,
                             save_path_pdf: str = "fig_return_distribution.pdf",
                             save_path_png: str | None = None) -> None:
    """
    日次リターン分布：
      左：ヒストグラム + 正規分布フィット
      右：QQプロット
    論文用の単独図
    """
    r = df["Return"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 左：ヒストグラム + 正規分布 ---
    ax0 = axes[0]
    ax0.hist(r, bins=80, density=True, alpha=0.7,
             edgecolor="white", linewidth=0.5,
             label="実績分布")

    x = np.linspace(r.min(), r.max(), 300)
    ax0.plot(x, norm.pdf(x, r.mean(), r.std()),
             linewidth=2, label="正規分布（同平均・分散）")

    ax0.set_title("日次リターン分布と正規近似", fontsize=12, fontweight="bold")
    ax0.set_xlabel("日次対数リターン", fontsize=10)
    ax0.set_ylabel("確率密度", fontsize=10)
    ax0.legend(fontsize=9)
    ax0.grid(alpha=0.3)

    textstr = f"歪度: {skew(r):.3f}\n超過尖度: {kurtosis(r, fisher=True):.3f}"
    ax0.text(0.97, 0.97, textstr,
             transform=ax0.transAxes,
             fontsize=9,
             verticalalignment="top",
             horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # --- 右：QQプロット ---
    ax1 = axes[1]
    probplot(r, dist="norm", plot=ax1)
    ax1.set_title("QQプロット（正規分布との比較）", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)

    # マーカーと基準線を整える
    line_data = ax1.get_lines()[0]
    line_ref = ax1.get_lines()[1]
    line_data.set_markerfacecolor("tab:blue")
    line_data.set_markeredgecolor("tab:blue")
    line_data.set_markersize(3)
    line_ref.set_linewidth(1.5)

    plt.tight_layout()

    plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"保存: {save_path_pdf}")

    if save_path_png is not None:
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"保存: {save_path_png}")

    plt.show()


# ----------------------------------------------
# 7. 可視化：リターンタイムライン＋イベント
# ----------------------------------------------
def plot_event_timeline(df: pd.DataFrame,
                        save_path_pdf: str = "fig_event_timeline.pdf",
                        save_path_png: str | None = None) -> None:
    """
    日次リターンのタイムライン + 共通イベントセット
    （正のリターン/負のリターンを色分け）
    """
    r = df["Return"]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.fill_between(r.index, 0, r,
                    where=(r >= 0),
                    alpha=0.6, label="正のリターン")
    ax.fill_between(r.index, 0, r,
                    where=(r < 0),
                    alpha=0.6, label="負のリターン")

    y_min = r.min()
    y_max = r.max()

    for ev in EVENTS:
        ts = pd.Timestamp(ev["date"])
        if ts < r.index.min() or ts > r.index.max():
            continue

        color = "red" if ev["region"] == "global" else "blue"
        linestyle = "--" if ev["region"] == "global" else ":"

        ax.axvline(ts, color=color, ls=linestyle, alpha=0.9, linewidth=1.0)

        if ev["region"] == "global":
            y_text = y_max * 0.9
            va = "top"
        else:
            y_text = y_min * 0.9
            va = "bottom"

        ax.text(ts, y_text, ev["name"],
                rotation=90,
                fontsize=7,
                color=color,
                ha="right",
                va=va)

    ax.set_title("日次リターンと主要イベントのタイムライン", fontsize=14, fontweight="bold")
    ax.set_xlabel("日付", fontsize=11)
    ax.set_ylabel("日次対数リターン", fontsize=11)
    ax.set_xlim(r.index.min(), r.index.max())
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(alpha=0.3, linestyle="-")

    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"保存: {save_path_pdf}")

    if save_path_png is not None:
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"保存: {save_path_png}")

    plt.show()


# ----------------------------------------------
# 8. メイン処理
# ----------------------------------------------
def main():
    """メイン実行関数"""
    print("\n" + "=" * 60)
    print("卒論第1段階：日経225専用データセット作成")
    print("=" * 60)

    # データ取得
    df = download_nikkei()

    # リターン計算
    df = calc_returns(df)

    # 統計量計算
    stats = compute_statistics(df)

    # CSV保存
    print("\n" + "-" * 50)
    print("CSVファイル保存")
    print("-" * 50)

    df[['Nikkei225']].to_csv("nikkei_prices.csv")
    print("保存: nikkei_prices.csv")

    df[['Return']].to_csv("nikkei_returns.csv")
    print("保存: nikkei_returns.csv")

    stats.to_csv("nikkei_statistics.csv", index=False)
    print("保存: nikkei_statistics.csv")

    # 可視化（図ごとに個別PDF＋任意でPNG）
    print("\n" + "-" * 50)
    print("可視化ファイル生成")
    print("-" * 50)

    plot_price(df,
               save_path_pdf="fig_price.pdf",
               save_path_png="fig_price.png")

    plot_return_distribution(df,
               save_path_pdf="fig_return_distribution.pdf",
               save_path_png="fig_return_distribution.png")

    plot_event_timeline(df,
               save_path_pdf="fig_event_timeline.pdf",
               save_path_png="fig_event_timeline.png")

    print("\n" + "=" * 60)
    print("✅ 日経225専用データセット作成 完了")
    print("=" * 60)
    print("\n生成ファイル一覧:")
    print("  📊 CSVデータ:")
    print("     - nikkei_prices.csv")
    print("     - nikkei_returns.csv")
    print("     - nikkei_statistics.csv")
    print("  📈 可視化:")
    print("     - fig_price.pdf / .png")
    print("     - fig_return_distribution.pdf / .png")
    print("     - fig_event_timeline.pdf / .png")
    print("\n→ 次のステップ: EWMA推定（第2段階）へ")

    return df, stats


if __name__ == "__main__":
    df, stats = main()