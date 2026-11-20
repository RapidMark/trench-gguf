# Hy4-flash (hy_v4_flash): BF16 のチェックポイントから低ビット GGUF を作る

[English](README.md)

Tencent の [Hy4-flash](https://huggingface.co/tencent/Hy4-flash)（`HYV4FlashForCausalLM`、2190 億パラメータ、42 層、
エキスパート 512 個のうち 16 個を使う）は、本家 llama.cpp にはまだ入っていません。`patches/llama-hy4.diff` が
trench-gguf 専用の llama.cpp のコピーにこれを足します（`./setup.sh` が `third_party/llama.cpp` と
`third_party/llama.cpp-gpu` に当てます）。これで、いつもの流れ（変換 → imatrix → 量子化 → 品質チェック）が使えます。
対象はテキストだけです（画像の部分と dfly ドラフターは入れていません）。

アーキテクチャ名・テンソル名・メタデータのキーは、公開されている hy_v4_flash の GGUF（peasantsmith/Hy4-Flash-Maya、
nace-ai/Hy4-flash-GGUF）と同じです。これらのファイルはそのまま読めます。

## パッチで足したもの

| 部分 | 内容 |
| --- | --- |
| `hy_v4_flash` モデル（`src/models/hy-v4-flash.cpp`） | 残差は 4 本のストリーム（hyper-connection）。ゲート付き MLA（q は 1024 の LoRA、共有の 512 次元潜在 + 64 次元の RoPE キー、ヘッドごとの学習済みシンク、出力の要素ごとのシグモイドゲート、attention 後の norm）。30 層はスライディングウィンドウ（64 ヘッド、512 キー、RoPE 1e4）、12 層はインデクサ付き（32 ヘッド、RoPE 1e7。キー 4 個ずつをゲート付き softmax でまとめ、クエリが 2048 キーより先を見るようになったら上位 511 グループ + 自分のグループだけを見る）。0 層目は密な FFN、残りは latent MoE（隠れ状態にシグモイドのルーター、エキスパートは 1536 次元への射影の上で動く、swiglu の clamp 10、共有エキスパート）。Engram は 1/11/21/31 層（2-gram と 3-gram のハッシュで 1 トークンあたり 96 次元の行を 16 本引く） |
| ggml の `F8_E4M3`（型 43） | ただの OCP E4M3 のバイト列。Maya の Engram テーブルの型です（番号は本家 ggml と同じ。unsloth の DeepSeek V4.1 の「型 43」は別物の MXFP8 で、そちらは `tools/requant_table.py` が読みます） |
| `llama-quantize` のルール | エキスパートはファイルの型どおり。`per_layer_token_embd`（Engram テーブル、634 億値）は既定で IQ4_NL、`--tensor-type per_layer_token_embd=<型>` で変えられます。ほかの行列は最低でも Q8_0。hyper-connection の重み、インデクサ、Engram のバイアスとスケールは F32 のまま |
| 変換スクリプト（`conversion/hy_v4_flash.py`） | `HYV4FlashForCausalLM` → `hy_v4_flash` の GGUF。kv_b を k_b / v_b に分け、gate_up を分け、Engram のキーと値の射影を 1 本にまとめ、norm_key × norm_query をゲートに畳み込み、4 層分の Engram テーブルを 1 本のテーブルへ流し込みます。ハッシュの素数は config と突き合わせます |
| `tools/teacher cmp` | 別のエンジンの全語彙の log-softmax ダンプ（Abyss の `STRATA_GLM_SCORE_DUMP`）に対して、語彙全体で KL を出します |

制限: インデクサのグループ選択（位置 2048 以降）は、KV のセルが位置 0, 1, 2, … の順に並んだ 1 本のシーケンスを前提に
しています（perplexity や imatrix のチャンク、新しく始めた生成）。2048 を越える複数シーケンスを 1 バッチに入れると
エラーで止まります。スコアがぴったり同じときは、前のグループを選びます。

## 確かめたこと

数字は [CHANGELOG](../../CHANGELOG.md) にまとめています。要点は次のとおりです。

- **Abyss との比較**（同じトークン列、Maya-XS と nace Q3_K_M、mix459 と 2,048 トークンの日本語）: 位置ごとの
  語彙全体の KL と、1 位のトークンが一致する割合です。

  | ファイル、テキスト | 位置の数 | KL 平均（中央値 / 99 パーセンタイル） | 1 位の一致 | nll llama.cpp / Abyss |
  | --- | --- | --- | --- | --- |
  | Maya-XS IQ2_XXS、mix459 | 457 | 0.118（0.0006 / 1.95） | 93.4% | 1.639 / 1.620 |
  | Maya-XS IQ2_XXS、ja2k | 2046 | 0.106（0.022 / 1.68） | 88.4% | 4.116 / 4.125 |
  | nace Q3_K_M、mix459 | 457 | 0.083（0.0002 / 1.72） | 93.7% | 1.655 / 1.734 |
  | nace Q3_K_M、ja2k | 2046 | 0.062（0.011 / 0.87） | 91.8% | 3.832 / 3.805 |

  ゼロにはなりません。2 つのエンジンは同じ計算を違う順番と精度でしています（Abyss は自前のカーネルで 1 トークンずつ、
  llama.cpp は 512 トークンずつのバッチで KV は F16）。2 bit では、ルーターのスコアが拮抗したところでエキスパートが
  入れ替わります。nll はどちらか一方がいつも低いわけではありません。mix459 の 0〜136 番目（日本語の部分）はよく
  一致し（KL 平均 0.011 / 0.008）、そのあとの英語とコードのほうがずれます（0.16 / 0.11）。
- **numpy のリファレンス（Abyss の `ref/hy4.py`）との比較**: チェックポイントと同じ構成の小さな乱数モデル（33 層、
  Engram 4 層、インデクサが 16 キーから選ぶ）で、16 トークンでは KL 0.000002。60 トークンでは中央値 0.000001、
  最大 0.003 でした。最大のところは、2 つのエキスパートのルータースコアの差が 7e-5 しかなく、足し算の順番の違いで
  もう一方が選ばれた位置です。
- **変換スクリプト**: 本物と同じ名前と形の合成チェックポイントは変換でき、読み込めます。まとめた・分けたテンソルは
  すべて期待どおりの値とビット単位で一致しました。変換のやり方そのものは、本物のチェックポイントの 1 層目と 11 層目を
  範囲指定で読み、nace の GGUF と比べて確かめています（norm と F32 のテンソルは完全一致、Q8_0 のテンソルは相関
  0.99999、エキスパートの gate / up / down は Q3_K に対して 0.988。ほかの並べ方だと 0.00）。
  **440 GB のチェックポイント全体ではまだ動かしていません。**

## 手順（`run.sh`）

| 手順 | ディスク | 時間 |
| --- | --- | --- |
| download | 440.5 GB（index.json の total_size） | 回線しだい（1 Gbps で約 1 時間） |
| convert（`--outtype bf16`） | 約 438 GB（2190 億パラメータ × 2 バイト。うち 0.4 GB は F32） | 未計測: 440 GB を読んで 438 GB を書く |
| calib | 約 2 MB | 数分（eaddario/imatrix-calibration をダウンロード） |
| imatrix（GPU、エキスパートはホストから） | 約 1 GB | BF16 では未計測。Maya-XS での速さは CHANGELOG に |
| quantize（エキスパート IQ2_XXS、テーブル IQ4_NL、残り Q8_0） | 約 82 GB: エキスパート 1486 億 × 2.06 bit = 38.3 GB、テーブル 634 億 × 4.5 bit = 35.6 GB、残り 69 億 × 8.5 bit = 7.3 GB | 未計測 |

Engram テーブルは行を引くだけで掛け算に使わないので、imatrix の統計が付きません。そのため imatrix のいらない型に
しています。`run.sh` の日本語向けの `--tensor-type` の引き上げは出発点で、測って決めたものではありません。
