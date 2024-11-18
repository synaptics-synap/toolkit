# Copying SyNAP Models to Board

| Implementation |
|----------------|
| [tools/copy.py](/tools/copy.py) |

This is a guide on copying SyNAP models to a Astra board over SSH. ADB support is planned for the future.

## Copy options
To copy models, run `python -m tools.copy`. The following options are available:
- `--serial`: The device serial for ADB. By default ADB will use the first connected device it finds, use this to specify a different device.
- `--board_ip`: The IP address of the board to copy models to. If provided, SSH will be used instead of ADB for copying.
- `--convert_dir`: Directory containing the SyNAP model folders. Default is `models/converted`.
- `--copy_dir`: The directory to copy models into on the board. Will be created if it doesn't exist. Default is `/home/root/models`.
- `--all | --latest | --models`: Select which models to copy from `--copy_dir`. Only one of these options may be specified at a time.
  - `--all`: Copy all models.
  - `--latest`: Copy the most recently converted model.
  - `--models NAME [NAME, ...]`: Copy all models corresponding to `NAME`s, which can be model filenames or a singular glob pattern.

A summarized version of this information is available via `python -m tools.convert --help`.

> [!NOTE]
> This tool is intended to be used in conjunction with [`tools.convert`](/tools/docs/convert.md). As such, model selection via `--models` is somewhat dependent on the format of the converted model filenames produced by the convert script.

## Copy examples
1. Copy all models:
```sh
python -m tools.copy \
  --all
```
2. Copy only models with 224x224 input size, over SSH:
```sh
python -m tools.copy \
  --models *224x224* \
  --board_ip <IP address>
```
3. Copy a model to a different directory:
```sh
python -m tools.copy \
  --latest \
  --copy_dir /tmp
```