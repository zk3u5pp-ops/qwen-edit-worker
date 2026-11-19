# Qwen-Image-Edit-2511 RunPod ワーカー（FP8/GGUF 修正版）

conalmullan/video-toolkit-qwen-edit を元にした修正フォーク。元イメージの handler は
`use_fp8` を無視して常に BF16 モデル（57.7GB）を読み込むため、24GB GPU では
OOM で動かなかった。このフォークでは量子化を正しく実装し、RTX 4090（24GB）で
動作する。

## 元イメージからの修正点

- **本物の FP8**：torchao の FP8 weight-only 量子化（diffusers `TorchAoConfig`）で
  transformer を約20.5GB にする。`use_fp8` が実際に効く。
- **GGUF バックエンド**：unsloth の GGUF transformer（Q6_K 17.95GB）も選択可能。
  実働例のある確実なフォールバック。
- **Lightning LoRA**：公式の 4ステップ（または 8ステップ）LoRA を読み込み、
  必須の shift=3 スケジューラと `true_cfg_scale=1.0` を設定。元ハンドラーは
  LoRA なしで 4ステップを回していたため画質が悪かった。
- **CPU オフロード**：16.6GB のテキストエンコーダーを CPU へ逃がし、拡散中は
  transformer のみを VRAM に常駐させる。
- **入力サイズ制限**：長辺 1024px を超える画像は自動で縮小し、VRAM を守る。

## ビルドと公開

GitHub Actions が `main` への push ごとにイメージをビルドし、GHCR に公開する。

1. このリポジトリを `qwen-edit-worker` という名前で作成し、ファイルを push する。
2. Actions の `Build and push worker image` が走り、
   `ghcr.io/<owner>/qwen-edit-worker:latest` が作られる。
3. **GHCR パッケージを公開設定にする**（デフォルトは非公開）。
   GitHub → プロフィール → Packages → qwen-edit-worker → Package settings →
   Danger Zone → Change visibility → Public。

## RunPod エンドポイント設定

| 項目 | 推奨値 |
|---|---|
| Container Image | `ghcr.io/<owner>/qwen-edit-worker:latest` |
| GPU | ADA_24（RTX 4090）のみ。torchao FP8 は CC 8.9 以上が必要 |
| Container Disk | 80GB（両バックエンドを使うなら 120GB） |
| Init Timeout | 1800秒（torchao 初回は 3600秒を推奨） |
| Max Workers | 1〜4 |
| Idle Timeout | 5秒 |
| 環境変数 | なし（`WARMUP_BACKEND=none` で起動時プリロードを無効化できる） |

起動時のプリロードは既定で `torchao_fp8`。初回は約58GBのダウンロード＋
CPU 量子化が走る。GGUF のみ使う場合は `WARMUP_BACKEND=gguf` にすると
約35GB で済み、起動が速い。

## 入出力

入力（`POST /run` の `input`）：

| フィールド | 必須 | 説明 |
|---|---|---|
| `image_base64` | ○ | 元画像（base64） |
| `images_base64` | | 参考画像（最大2枚） |
| `prompt` | ○ | 編集指示 |
| `negative_prompt` | | ネガティブプロンプト |
| `backend` | | `torchao_fp8`（既定）または `gguf` |
| `use_fp8` | | 旧フラグ。true→torchao_fp8、false→gguf |
| `gguf_quant` | | `Q6_K`（既定）、`Q8_0`、`Q4_K_M` |
| `lightning_steps` | | `4`（既定）または `8`（LoRA とステップ数を切替） |
| `seed` | | シード（未指定はランダム） |
| `num_inference_steps` | | 無視される（Lightning LoRA が固定） |
| `guidance_scale` | | 無視される（Qwen は非対応） |
| `auto_resize` | | 長辺 1024px 超の自動縮小（既定 true） |

出力：`edited_image_base64`、`seed`、`inference_time_ms`、`image_size`、
`num_inference_steps`、`backend`、`use_fp8`。

## トラブルシューティング

- **OOM が出る**：`backend: "gguf"` で試す。Q6_K は 17.95GB で余裕がある。
- **ワーカーが起動しない（init timeout）**：Init Timeout を 3600秒に上げるか、
  `WARMUP_BACKEND=gguf` にする。
- **画質が悪い／グリッド状の模様が出る**：torchao FP8＋LoRA の組み合わせは
  未検証。GGUF バックエンドで比較すること。
- **ディスク不足**：両バックエンドを同じワーカーで使うと計約95GB になる。
  Container Disk を 120GB に上げること。
