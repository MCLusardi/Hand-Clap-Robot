# Hand-Clap-Robot

## TODO:
- Add install instructions
- ?


## Installation
### Clone Repo
```
git clone git@github.com:MCLusardi/Hand-Clap-Robot.git --recurse-submodules
```

### Make virtual env then source
```
source .venv/bin/activate
```

### Install
```
export PYTHONPATH=.
```

```
pip install -r requirements.txt
```

```
cp temp/pyproject.toml ur_ikfast/pyproject.toml
```

```
cd ur_ikfast && pip install -e .
```
