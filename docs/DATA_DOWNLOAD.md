# データセット選定と再開可能なダウンロード

## 推奨する導入順

容量と研究上の役割を考えると、最初から全データを揃える必要はありません。

| 段階 | データ | 役割 | 取得量の目安 | 初回の手動操作 |
| --- | --- | --- | ---: | --- |
| 0 | DSEC 1 sequence | downloader・前処理・時刻窓の確認 | sequence依存 | なし |
| 1 | M3ED Phase 1 train+val | 異なる解像度を含む事前学習の成立確認 | 20,262,591,423 bytes | なし |
| 2 | DSEC Detection train+val | driving domainと下流検出 | train全体131.9 GB相当 + extra val 21.9 GB | なし |
| 3 | Gen1 | 304×240での検出baseline | 200 GB圧縮、750 GB展開 | **公式フォーム・CAPTCHA** |
| 4 | M3ED test / DSEC test | 最終評価 | M3ED候補137.1 GB、DSEC約36.0 GB | なし |
| 5 | Prophesee 1Mpx | 高解像度transferと検出 | 1.6 TB圧縮、3.5 TB展開 | **公式フォーム・メール確認** |

主な事前学習データはM3EDとDSEC、下流の成立確認はDSEC DetectionとGen1、
1Mpxは最後のscale-upとするのが現実的です。M3EDはcar/spot/falconを一度に
取得せず、まず同じcar/urban/day条件のtrain/val 1本ずつから始めます。

Gen1は1Mpxより小さいものの、全量では軽量ではありません。公式にはtrain 6、
val 2、test 2の独立archiveなので、まず1 archiveだけ取得できます。1Mpxは
archive数・固定ファイル名・publisher checksumが公開されていないため、容量に
十分な余裕ができるまで保留します。

## downloaderの共通仕様

download script自体はBash、`curl`、Python標準ライブラリだけを使います。
PyTorch、h5py、hdf5plugin、仮想環境はまだ不要です。

```bash
chmod +x scripts/download/*.sh scripts/download/archive_tool.py
```

各scriptは次を共通に行います。

- 未完了ファイルを`*.part`として残し、同じcommandでHTTP Range resume
- resume前にstrong ETag、配布元checksum、または既知のpublisher SHA-256を
  必須とし、Content-LengthやLast-Modifiedもsidecarと照合
- 保護URLをsidecarやログへ保存せず、`curl`のprocess引数にも直接置かない
- HTMLのlogin/error pageをデータとして受理しない
- ZIP/TARのintegrity検査、HDF5 signature、byte数、ローカルSHA-256を記録
- 検証後だけatomic renameし、完成済みファイルは再取得しない
- archive展開は各memberを一時ファイルから置換し、完了記録から再開
- path traversal、symlink、特殊memberを含むarchiveを拒否
- source archive・source HDF5を自動削除しない
- 同じ出力への並行書込みをatomic lockで拒否する

publisherが公式SHA-256を提供していない場合、記録されるローカルSHA-256は
再実行時の同一性確認用であり、publisher真正性の証明ではありません。remote identityが
変わった場合は既存`.part`へ追記せず停止します。署名URLが失効した場合は同じローカル
filenameに対する新しいURLへ差し替えて再実行してください。
通常の中断ではlockを自動解除します。`kill -9`や電源断後に`*.lock`が残った場合は、
同じdownload processが動いていないことを確認してから、そのlock directoryだけを
手動で除いて再実行します。

展開は既定で無効です。archiveと展開後データを同時に置く空き容量を確認してから
`--extract`を付けます。converter/HDF5検証が終わるまでarchiveは残してください。

## DSEC / DSEC-Detection

DSECは認証もGUIも不要で、公式HTTPS URLから直接取得できます。最初の1 sequenceは
次のcommandです。Detection label全60系列分は約4.7 MBなので既定で併せて取得します。
公式ページにはデータ自体の再配布条件が明記されていないため、utility codeのlicenseを
データの再配布許諾とは解釈しないでください。

```bash
bash scripts/download/download_dsec.sh \
  --root /datasets/downloads/dsec \
  --profile custom \
  --physical-split train \
  --sequence-list configs/datasets/dsec_phase1_smoke.txt
```

同じcommandへ`--extract`を追加すると、完成済みdownloadを飛ばして
`/datasets/downloads/dsec/raw`へ安全に展開します。

```bash
bash scripts/download/download_dsec.sh \
  --root /datasets/downloads/dsec \
  --profile custom \
  --physical-split train \
  --sequence-list configs/datasets/dsec_phase1_smoke.txt \
  --extract
```

成立確認後は論理splitごとのprofileを使います。

```bash
# 元DSEC train 41本。sequence単位なので途中停止・再開しやすい。
bash scripts/download/download_dsec.sh \
  --root /datasets/downloads/dsec --profile detection-train

# 論理val 6本。物理配置はtrain/だが、このextra ZIPだけに入っている。
bash scripts/download/download_dsec.sh \
  --root /datasets/downloads/dsec --profile detection-val

# 最終段階: 元test 12本 + extra-only thun_02_a。
bash scripts/download/download_dsec.sh \
  --root /datasets/downloads/dsec --profile detection-test
```

raw eventとDetection bboxはいずれもdistorted event-camera座標なので、この用途では
calibrationは不要です。flow・disparity・semanticも扱う段階でだけ
`--include-calibration`を付けます。valの`zurich_city_16_a`〜`21_a`は物理的には
`train/`、testの`thun_02_a`だけはextra配布物です。scriptはこの差をprofile内で処理します。

## M3ED

M3EDも認証・GUI不要です。公式S3からprocessed `*_data.h5`だけを取得し、動画、
ROS bag、depth、pose、point cloudはdownloadしません。公式`dataset_list.yaml`は
再現性のためcommitを固定して保存し、`is_test_file`をdownload前に照合します。

```bash
bash scripts/download/download_m3ed.sh \
  --root /datasets/downloads/m3ed \
  --split train \
  --sequence-list configs/datasets/m3ed_phase1_train.txt

bash scripts/download/download_m3ed.sh \
  --root /datasets/downloads/m3ed \
  --split val \
  --sequence-list configs/datasets/m3ed_phase1_val.txt
```

Phase 1の選定は次の2本です。

- train: `car_urban_day_penno_small_loop`、13,798,490,482 bytes
- val: `car_urban_day_horse`、6,464,100,941 bytes

どちらも公式non-testです。M3EDには公式valがないため、valはnon-test recordingから
こちらで固定したhold-outです。test候補`car_urban_day_ucity_big_loop`は
137,070,365,939 bytesと大きいため、Phase 1では取得しません。必要になった時点で
`configs/datasets/m3ed_phase2_test.txt`を使います。

```bash
bash scripts/download/download_m3ed.sh \
  --root /datasets/downloads/m3ed \
  --split test \
  --sequence-list configs/datasets/m3ed_phase2_test.txt
```

## Gen1

Gen1の初回URL取得はCLIだけでは完結しません。公式ページからZohoフォームを開き、
氏名・メール・所属等を入力し、利用条件への同意とCAPTCHAを手動で完了してください。
利用条件は研究者本人が確認する必要があります。

- 公式ページ: https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/
- 公式フォーム: https://forms.zohopublic.com/itdesk175/form/DatasetAtisAutomotiveDetection/formperma/c8fMk4X9Y2P5f-H8kXUHULiDXyGp4a04i027OnpQePQ

URLをコピーできる場合は、exampleをignore対象のprivate fileへ複製し、
1行ごとに`filename URL [sha256|-] [bytes|-]`を書きます。URLをshell scriptへ埋め込んだり
commitしたりしないでください。URL fileはsplitごとに分けます。
scriptはarchiveを`ROOT/archives/<split>`、展開物を`ROOT/raw/<split>`へ置きます。
split directoryを持たない展開物は前処理での混入を防ぐため拒否します。

```bash
cp configs/download/gen1_urls.example.txt configs/download/gen1_train.private.urls
chmod 600 configs/download/gen1_train.private.urls
# configs/download/gen1_train.private.urlsを編集

bash scripts/download/download_gen1.sh \
  --root /datasets/downloads/gen1 \
  --split train \
  --url-file configs/download/gen1_train.private.urls
```

ブラウザから既にarchiveを取得した場合は、そのdirectoryをinboxとして検査・展開できます。
この場合、ブラウザ側downloadの再開機能はブラウザに依存します。

```bash
bash scripts/download/download_gen1.sh \
  --root /datasets/downloads/gen1 \
  --split train \
  --inbox /path/to/manual/gen1 \
  --extract
```

URL file方式ならscriptの`.part`再開を利用できます。まず1 archiveだけをURL fileへ記載し、
DATとbboxの対応、DAT headerに解像度があれば304×240との一致、前処理時の全event座標範囲を
確認してから増やしてください。全量ではarchiveと展開rawだけで約950 GBになるため、
canonical HDF5の作業領域も含めて1 TB超を想定します。

配布物がZIP/TAR以外だった場合は、7-Zip等で手動展開して明示的なsplit directoryへ置き、
依存なしの検査だけを実行できます。

```bash
bash scripts/download/download_gen1.sh \
  --root /datasets/downloads/gen1 \
  --split train \
  --extracted-root /datasets/downloads/gen1/raw/train
```

## Prophesee 1Mpx

1Mpxも公式フォーム、professional emailの検証、利用条件への同意、CAPTCHAを手動で
完了する必要があります。download以降の使い方はGen1と同じです。

- 公式ページ: https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/
- 公式フォーム: https://forms.zohopublic.com/itdesk175/form/Dataset1MegapixelAutomotiveDetection/formperma/m8gOxbwaLFXc2PaLpalHNXeKpq4Tdci1DL0Ynx8q_FE

```bash
cp configs/download/prophesee_1mpx_urls.example.txt \
  configs/download/1mpx_train.private.urls
chmod 600 configs/download/1mpx_train.private.urls
# configs/download/1mpx_train.private.urlsを編集

bash scripts/download/download_prophesee_1mpx.sh \
  --root /datasets/downloads/1mpx \
  --split train \
  --url-file configs/download/1mpx_train.private.urls
```

GUI取得済みなら`--split SPLIT --inbox DIRECTORY`を使い、空き容量を確認後に
`--extract`を付けます。全rawを圧縮archiveと同時保持するだけで約5.1 TBになるため、
canonical HDF5の作業領域まで
含めて6 TB超を想定してください。RVT等の第三者mirrorは便利でも、公式フォームの
利用条件と取得経路を迂回するおそれがあるため、この実装では自動利用しません。

## 前処理への接続

DSECとProphesee系は`ROOT/raw`、M3EDは同じく`ROOT/raw`を
`window-jepa-preprocess --input`へ渡します。具体的な時刻補正、split、解像度、
Zstd HDF5変換は[DATA_PREPROCESSING.md](DATA_PREPROCESSING.md)を参照してください。

容量が厳しい場合は、1 sequenceまたは1 archiveごとに

1. downloadとintegrity検査
2. 展開
3. canonical Zstd HDF5へ変換
4. manifestとrandom sliceを別環境で検証
5. source/archiveの保管または削除を**人が判断**

の順に進めます。scriptは削除を自動化しません。

## 公式情報

- [DSEC Downloads](https://dsec.ifi.uzh.ch/dsec-datasets/download/)
- [DSEC Data Format](https://dsec.ifi.uzh.ch/data-format/)
- [DSEC-Detection](https://dsec.ifi.uzh.ch/dsec-detection/)
- [M3ED Download](https://m3ed.io/download/)
- [M3ED official dataset list（固定commit）](https://github.com/daniilidis-group/m3ed/blob/df739f20fba41ac6da8c22f4260c305875e391ed/dataset_list.yaml)
- [Prophesee dataset formats and sizes](https://docs.prophesee.ai/stable/datasets.html)
- [Prophesee automotive dataset toolbox](https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox)
