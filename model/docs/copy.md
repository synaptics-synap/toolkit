# Copying SyNAP Models to Board

| Implementation |
|----------------|
| [model/copy.py](/model/copy.py) |

This is a guide on copying SyNAP models to a Astra board over SSH. ADB support is planned for the future.

## Copy options
To copy models, run `python -m model.copy`. The following options are available:
- `--board_ip`: The IP address of the board to copy models to.
- `--convert_dir`: Directory containing the SyNAP model folders. Default is `models/converted`.
- `--copy_dir`: The directory to copy models into on the board. Will be created if it doesn't exist. Default is `/home/root/models`.
- `--all | --latest | --models`: Select which models to copy from `--copy_dir`. Only one of these options may be specified at a time.
  - `--all`: Copy all models.
  - `--latest`: Copy the most recently converted model.
  - `--models NAME [NAME, ...]`: Copy all models corresponding to `NAME`s, which can be model filenames or a singular glob pattern.

A summarized version of this information is available via `python -m model.convert --help`.

> [!NOTE]
> This tool is intended to be used in conjunction with [`model.convert`](/model/docs/convert.md). As such, model selection via `--models` is somewhat dependent on the format of the converted model filenames produced by the convert script.

## Copy examples
1. Copy all models:
```
python -m model.copy --all --board_ip 10.3.10.78
```
2. Copy only models with 224x224 input size:
```
python -m model.copy --models *224x224* --board_ip 10.3.10.78
```
3. Copy a model to a different directory:
```
python -m model.copy --latest --board_ip 10.3.10.78 --copy_dir /tmp
```