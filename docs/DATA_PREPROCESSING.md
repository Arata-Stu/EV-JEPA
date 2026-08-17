# DSEC・M3ED・Gen1・Gen4/1Mpxの前処理

公式配布元、段階的なデータセット選定、再開可能なdownload script、Gen1/1Mpxで
必要な手動フォーム操作は[DATA_DOWNLOAD.md](DATA_DOWNLOAD.md)を先に参照してください。

## 方針

Window-JEPAでは蓄積時間を学習時に選ぶため、5 msや40 msのvoxelを前処理で
固定保存しません。前処理の出力は、時刻順の連続event streamと1 ms単位の
coarse indexだけです。

```text
sequence.h5
├── events/
│   ├── x             uint16[N]
│   ├── y             uint16[N]
│   ├── t_us          uint32[N] または uint64[N]
│   └── polarity      uint8[N]   # 0=OFF, 1=ON
└── index/
    └── ms_to_event_idx uint32[K] または uint64[K]
```

各配列は`hdf5plugin.Zstd(clevel=5)`、shuffle、Fletcher32 checksumを使います。
HDF5 chunkは既定262,144 eventsです。約71分以内のsequenceでは時刻を先頭event
基準の`uint32` microsecondへ変換し、それより長い場合は自動的に`uint64`を
使います。元時計の先頭時刻はroot属性`source_time_origin_us`へ保存するので、
ラベル時刻は次のように内部時刻へ戻せます。

```python
label_t_internal_us = label_t_source_us - source_time_origin_us
```

出力は一時ファイルへ書き、完了属性とindexを確定してからatomic renameします。
各chunkの確定位置も保存するため、M3EDのような巨大sequenceで中断しても、同じ
commandを再実行すれば互換性を確認して`.partial`から再開します。入力のsize・mtime、
変換設定、split、解像度が変わっていれば誤再開せず停止します。入力ファイルを削除する
処理はありません。

## 想定データセット

| dataset | native解像度 | 推奨前処理 | 主な用途 |
| --- | ---: | --- | --- |
| DSEC | 640×480 | left、倍率1、distortedのまま | 事前学習、Detection |
| M3ED | 1280×720 | leftから開始、整数2分の1 | 大規模・異種環境事前学習 |
| RVT Gen4 / Prophesee 1Mpx | 1280×720 | 整数2分の1 | 高解像度Detection |
| RVT Gen1 / Prophesee Gen1 | 304×240 | 倍率1 | 小規模な成立確認 |

1280×720を640×480へ非等方変形しません。M3EDと1Mpxは`x//=2, y//=2`で
640×360にし、DSECは640×480のまま保持します。学習時に全datasetから同じ
224×224 cropを取得します。空間downsample後もevent行は間引かず、同じpixelへ
重なったeventを別eventとして保持します。したがって時間密度と極性統計は残ります。

このconverterはdenseな1280×720 frameを生成せず、DATをevent chunk単位で処理します。
したがってnative解像度そのものよりevent数が前処理時間、I/O、容量を支配します。
空間downsampleだけではevent件数は減らないため、容量削減の中心は次の3点です。

- M3EDから画像・LiDAR・IMUを複製せず、使用するevent cameraだけを抽出する
- 1Mpx DATを`uint16/uint32/uint8`中心のSoAへ変換してZstd圧縮する
- right cameraが不要な段階ではleftだけを変換する

特に1Mpx全量を一度に変換せず、固定したrecording listでまず10--20%だけをportable
bundleへ変換するのが安全です。factor 2は容量を4分の1にする設定ではありません。

event strideや固定時間binによる間引きは、蓄積時間研究で重要なevent countを変える
ため実装していません。

## 導入

```bash
python3.11 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[hdf5,dev]'
```

Zstd filterはPython process内で`hdf5plugin`をimportして登録する必要があります。
本リポジトリのconverterと`H5EventStore`はHDF5を開く前に登録します。

## DSEC

DSEC公式`events.h5`は既にBloscのZSTD codecで圧縮され、構造は
`/events/{x,y,t,p}`, `/ms_to_idx`, `/t_offset`です。converterは
`t_abs=t+t_offset`へ直してから共通の相対時刻へ変換し、元の`t_offset`も属性へ
残します。

```bash
window-jepa-preprocess \
  --dataset dsec \
  --input /datasets/DSEC/train \
  --output-root /datasets/evjepa/dsec/train \
  --manifest /datasets/evjepa/manifests/dsec_train.jsonl \
  --split train \
  --camera left \
  --sequence-list configs/datasets/dsec_detection_train.txt \
  --spatial-downsample 1
```

`--sequence-list`は1行1sequence名のplain textです。DSEC-Detectionの論理val
sequenceは物理的にはtrain directory内にあるため、親directory名からsplitを
推測しないでください。train/val/testを別々に実行します。

DSEC raw eventはdistorted座標です。Detection labelもこのevent viewに対応するため、
事前学習とDetectionではraw座標を保持します。flow/disparity/semanticはrectified
座標なので、下流adapterで公式rectify mapを適用します。semantic labelはさらに
下40行を持たない640×440です。rectification済みeventを同じmanifestへ混ぜません。

DSEC入力自体がZSTD圧縮済みなので、再packで必ず小さくなるとは限りません。変換中は
入力と出力の両方が必要です。空き容量が厳しい場合はsequenceごとに出力を検証して
から元データを手動でarchiveしてください。converterは元データを削除しません。
DSECはラベル座標との対応を守るため、converterでは`--spatial-downsample 1`だけを
許可します。

## M3ED

M3ED processed HDF5の`/prophesee/{left,right}/{x,y,t,p}`をstreamingで読みます。
`/prophesee/{camera}/calib/resolution`があれば解像度をそこから取得し、なければ
公式の1280×720を使用します。`t`は公式converterがsensor時刻のskew/offsetを補正した
同期global clockのmicrosecond値であり、その時計情報もH5属性へ残します。
`--width/--height`は解像度metadataが欠落した入力にだけ使えます。公式calibrationや
DAT headerが存在する場合、一致しない上書き値は座標系破損として拒否します。
M3EDのdecoded event座標もこの段階ではrectifyせず、`distorted`として保持します。
depth・flowなどと組み合わせる場合は、別途保持した公式calibrationによる変換を下流側で
適用します。

```bash
window-jepa-preprocess \
  --dataset m3ed \
  --input /datasets/M3ED \
  --output-root /datasets/evjepa/m3ed/train \
  --manifest /datasets/evjepa/manifests/m3ed_train.jsonl \
  --split train \
  --camera left \
  --m3ed-dataset-list /path/to/m3ed/dataset_list.yaml \
  --sequence-list /path/to/m3ed_train_sequences.txt \
  --spatial-downsample 2
```

公式`dataset_list.yaml`の`is_test_file`を必ず照合します。M3EDには公式validation
splitがないため、非test系列からsession単位のvalidation listを自分で固定し、残りを
train listにします。このためtrain/valはいずれも`--sequence-list`必須です。左右を両方
使う場合は、同じsource recordingの左右を必ず同じsplitに置きます。まず
car/urban/daylightを小さなsequence listで確認し、その後spot/falcon、night、
off-roadへ広げます。

train/val/testの変換後に3 manifestを`window-jepa-merge-manifests`で一度統合すると、
同一source recordingが複数splitへ入っていないことも検査できます。統合manifestは
各行にsplitを保持するため、そのままloaderからsplit別に選択できます。

## RVT original-event HDF5（Gen1 / Gen4、推奨）

RVT配布版は、Gen4が`*_td.h5`、Gen1が`*_td.dat.h5`で、どちらも
`/events/{x,y,p,t}`に連続event列を保持しています。RVTの固定representation済みtarは
使用しません。converterは入力をchunk読みし、timestampの全体非減少、整数dtype、
`x/y/p/t`の同長、極性、座標範囲、解像度を検査します。RVT実装のように逆行timestampを
扱うため、RVT Gen1/Gen4入力に限ってevent順を維持したrunning maximum補正を適用します。
補正件数と最大逆行量は進捗ログ、出力HDF5属性、manifestへ記録します。DSEC/M3EDなど
他の入力ではtimestamp逆行を引き続きエラーにします。補正後の最大時刻がsource末尾の
生timestampを超える場合は、RVT入力だけ実効durationをその差分だけ延長し、
`timestamp_duration_extension_us`として記録します。

既存Gen4の最初の1 recordingを、出力を書かずに検査・計画表示する例です。

```bash
cd /home/aten-22/project/research/EV-JEPA
source env/bin/activate

window-jepa-preprocess \
  --dataset gen4 \
  --input /mnt/ssd-4tb/dataset/gen4 \
  --output-root /mnt/ssd-4tb/dataset/evjepa/gen4/events/train \
  --bbox-output-root /mnt/ssd-4tb/dataset/evjepa/gen4/labels/train \
  --manifest /mnt/ssd-4tb/dataset/evjepa/gen4/manifests/train.jsonl \
  --split train \
  --spatial-downsample 2 \
  --limit 1 \
  --plan-only
```

計画が正しければ`--plan-only`と`--limit 1`を外し、`--skip-existing`と
`--merge-manifest`を付けて変換します。val/testは出力directory、manifest、`--split`を
それぞれ変えて実行してください。変換中は既定10秒間隔で、event進捗率、速度、ETA、
timestamp補正統計をJSONで表示します。`--progress-interval-seconds`で間隔を変更できます。

```bash
window-jepa-preprocess \
  --dataset gen4 \
  --input /mnt/ssd-4tb/dataset/gen4 \
  --output-root /mnt/ssd-4tb/dataset/evjepa/gen4/events/train \
  --bbox-output-root /mnt/ssd-4tb/dataset/evjepa/gen4/labels/train \
  --manifest /mnt/ssd-4tb/dataset/evjepa/gen4/manifests/train.jsonl \
  --split train \
  --spatial-downsample 2 \
  --skip-existing \
  --merge-manifest
```

RVT Gen1はdownload rootの`raw`を入力にし、倍率1にします。

```bash
window-jepa-preprocess \
  --dataset gen1 \
  --input /mnt/ssd-4tb/dataset/gen1_rvt_h5/raw \
  --output-root /mnt/ssd-4tb/dataset/evjepa/gen1/events/train \
  --bbox-output-root /mnt/ssd-4tb/dataset/evjepa/gen1/labels/train \
  --manifest /mnt/ssd-4tb/dataset/evjepa/gen1/manifests/train.jsonl \
  --split train \
  --spatial-downsample 1 \
  --skip-existing \
  --merge-manifest
```

HDF5内に`/events/width,height`があれば公式解像度と照合し、無ければGen4は
1280×720、Gen1は304×240を使います。Gen4は`x//=2,y//=2`で640×360、Gen1は
304×240のままです。bboxはnative座標のまま別に保存し、manifestへ倍率を記録します。
既存データroot内の`_excluded_failed_validation`は再帰探索から明示的に除外します。

## Prophesee DAT（代替入力）

元のProphesee配布を使う場合は、Event2Dの8-byte DAT recordをchunk読みし、
32-bit timestamp wrapを展開します。
headerに解像度がない場合、1Mpxは1280×720、Gen1は304×240を使います。

1Mpxには誤って1280×720のまま全量変換しないための専用wrapperがあります。出力は
次のような、directoryごと移動できるbundleになります。

```text
1mpx_640x360/
├── events/{train,val,test}/*.h5
├── labels/{train,val,test}/*_bbox.npy
└── manifests/{train,val,test}.jsonl
```

manifest内のeventとbboxのpathは`manifests/`からの相対pathです。rawの`*_bbox.npy`は
1次元structured NPYであること、`t`/`ts`,`x`,`y`,`w`,`h`、有限値、正のboxサイズを
検査してから、座標やtimestampを書き換えずatomic copyします。公式Gen1/Gen4 raw label
にはframe外へ部分的または完全に出る既知boxがあるため、これを破損扱いにはしません。
`bbox_out_of_fov_count`と`bbox_requires_fov_clip`をmanifestへ記録し、検出adapter側で
native FOVへclipして幅または高さが0になったboxを除外してからfactorを適用します。
対象物がないclipの0件structured arrayは、有効なlabel fileとして保持します。
manifestにはnative/stored解像度、factor、`bbox_timestamp_reference`、
`bbox_timestamps_relative: false`、source時計を記録します。bbox timestampから
`source_time_origin_us`を引くとevent H5の内部時計になります。bundle全体を同じ構造の
まま別storageへ移動できます。

まず、変換対象DATのstem（例: `recording_name_td`）を1行1件で固定します。容量に
余裕がない段階ではtrain全体の10--20%を、昼夜・市街地・天候が偏らないように選び、
このlist自体も実験設定として保存してください。

```text
# /datasets/splits/1mpx_train_10pct.txt
recording_name_001_td
recording_name_014_td
```

書込み前の検査と計画表示:

```bash
bash scripts/preprocess/preprocess_prophesee_1mpx.sh \
  --python-bin env/bin/python \
  --input /datasets/1mpx/train \
  --bundle-root /datasets/evjepa/1mpx_640x360 \
  --split train \
  --sequence-list /datasets/splits/1mpx_train_10pct.txt \
  --plan-only
```

実変換は同じcommandから`--plan-only`だけを外します。wrapperはfactor 2、Zstd level
5、完了済みfileの検証付きskip、互換性のある`.partial`からの再開を固定しています。
同じcommandを再実行して構いません。同一出力を複数processが同時に更新しようとした
場合は、出力単位のadvisory lockで後発processを停止します。subsetへrecordingを段階的に
追加した場合も、wrapperはmanifest lockの内側で既存行を読んでartifactの存在と同一IDの
identityを検査してから新しい行をmergeするため、以前のclipをmanifestから落としません。

train/valでは対応するsibling `*_bbox.npy`が1件でもなければ、書込み開始前に停止します。
純粋なself-supervised事前学習としてlabelなしrecordingを意図的に使う場合だけ
`--self-supervised`を付けます。testでbboxが配布されていない場合はmanifestの
`bbox_path`を省略します。

factor 2は`x//=2, y//=2`で640×360へ座標を揃える設定で、event行とtimestampは一切
間引きません。このprojectの標準pipelineはeventをcropしてから224×224表現を作るため、
nativeとfactor 2でdense tensor/token数は同じです。またfactor 2の224 cropはnative
座標でより広い面積に相当し、含まれるevent数が増える場合もあります。主目的は
M3ED/Gen4間でstored gridを640×360へ揃えることと、同じ224×224 cropでより広い
native領域を扱うことです。DSECは640×480、Gen1は304×240のままで、角度FOVや
物体scaleまで同一にはなりません。H5容量や学習時間の4倍削減でもありません。
factor 4の320×180は標準224 cropより高さが小さいため使用しません。

全量変換の前に、同じ1--3 recordingを別outputへnative（factor 1）とfactor 2で変換し、
完了ログの`output_bytes`、実際のwindow内event数、DataLoader速度を実行環境で比較して
採用値を確定してください。native pilotは汎用CLIで行えます。

```bash
window-jepa-preprocess \
  --dataset prophesee_1mpx \
  --input /datasets/1mpx/train \
  --output-root /datasets/evjepa/pilot_native/events/train \
  --bbox-output-root /datasets/evjepa/pilot_native/labels/train \
  --manifest /datasets/evjepa/pilot_native/manifests/train.jsonl \
  --split train \
  --sequence-list /datasets/splits/1mpx_pilot.txt \
  --spatial-downsample 1 \
  --skip-existing
```

DAT版Gen1も`--dataset gen1 --spatial-downsample 1`です。Prophesee bboxはportable bundleへ
rawのままcopyされ、converterは下流labelを書き換えません。空間downsampleしたeventと
labelを組み合わせる場合は、下流adapterでbbox座標へ同じ倍率を適用し、bbox timestamp
からmanifestの`source_time_origin_us`を引いて内部時計へ揃えます。DSEC labelsと
calibrationも小さいため別途保持してください。
Gen4/1Mpx/Gen1をdataset rootから再帰探索する場合も、各event fileの最寄りの
`train`/`val`/`validation`/`test` directoryと`--split`を照合し、別splitのclipを
混入させません。

## manifestの統合と学習

```bash
window-jepa-merge-manifests \
  --output /datasets/evjepa/manifests/pretrain.jsonl \
  /datasets/evjepa/manifests/dsec_train.jsonl \
  /datasets/evjepa/manifests/m3ed_train.jsonl \
  /datasets/evjepa/manifests/1mpx_train.jsonl
```

`configs/pretrain/window_jepa_vits.yaml`の`data.manifest`を統合manifestへ変更します。
`sequence_sampling: dataset_balanced`では、最初にdatasetを一様選択し、その中から
sequenceを一様選択します。sequence数の多いdatasetだけが学習を支配するのを防ぎます。

## 安全な試行順序

最初は各datasetの1 sequenceだけを`--limit 1`で変換し、別環境でテストと短い
DataLoader確認を行ってください。再実行時は、既存の完了済みH5を検査して使う
`--skip-existing`か、atomicに置き換える`--overwrite`のどちらかを指定できます。
互換性のある未完了`.partial`は既定で自動再開します。破棄して最初から変換する場合だけ
`--overwrite --no-resume-partial`を併用します。完了ログには入力・出力byte数も表示
されるため、代表sequenceで容量と読出し速度を確認してから全量変換してください。
splitの異なるmanifestへ同じsequence IDを入れないでください。

## 仕様参照

- [DSEC Data Format](https://dsec.ifi.uzh.ch/data-format/)
- [DSEC-Detection utility / official split](https://github.com/uzh-rpg/dsec-det)
- [M3ED Data Files](https://m3ed.io/data_overview/datafiles/)
- [M3ED official 1 ms event loader](https://github.com/daniilidis-group/m3ed/blob/main/event_loading_by_ms.py)
- [Prophesee datasets and recording formats](https://docs.prophesee.ai/stable/datasets.html)
- [hdf5plugin Zstd usage](https://hdf5plugin.readthedocs.io/en/stable/usage.html#zstd)
