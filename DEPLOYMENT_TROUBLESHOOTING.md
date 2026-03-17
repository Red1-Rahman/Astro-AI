# 🚨 Streamlit Cloud Deployment Troubleshooting

## 🚀 **PRODUCTION SCIENTIFIC DEPLOYMENT**

**Status**: Deploying complete scientific stack with `21cmFAST`, `tools21cm`, `bagpipes`, `jwst_pipeline`

**Deployment Strategy**: Full scientific stack only - no minimal fallback  
**Expected Build Time**: 10-15 minutes (complete astronomical package suite)  
**System Packages**: 15 essential libraries including FFTW, GSL, CFITSIO  
**Python Packages**: Complete astronomical analysis capabilities with numpy 2.x compatibility

**Production-ready configuration** - all features enabled from deployment

---

## �🔥 **CRITICAL FIX: packages.txt Error**

**If you see errors like "Unable to locate package #" or "Unable to locate package GRAPHICS":**

**❌ PROBLEM**: Your `packages.txt` file contains comments that apt-get is trying to install as packages.

**✅ SOLUTION**: Use ONLY package names, NO comments in `packages.txt`.

**Current Fixed Version (Full Scientific):**

```txt
libfreetype-dev
libpng-dev
libhdf5-dev
libfftw3-dev
libgsl-dev
libcfitsio-dev
libopenblas-dev
liblapack-dev
build-essential
gcc
g++
gfortran
cmake
python3-dev
libx11-dev
```

**Emergency Reduced Version** (only if production deployment fails):

```txt
libfreetype-dev
libpng-dev
libhdf5-dev
```

## 🔥 **CRITICAL FIX: ModuleNotFoundError h5py** ⚠️ **RECURRING ISSUE**

**If you see "ModuleNotFoundError: No module named 'h5py'" or similar import errors:**

**❌ PROBLEM**: Streamlit Cloud is using `environment.yml` instead of optimized `requirements.txt`.

**⚠️ NOTE**: This issue can recur if `environment.yml` gets recreated. Always ensure it's completely removed.

**✅ SOLUTION**: Remove `environment.yml` file so Streamlit uses `requirements.txt`.

**Current Error Pattern:**

```
File "/mount/src/cam-sust2025/utils/data_handler.py", line 8, in <module>
    import h5py
ModuleNotFoundError: This app has encountered an error.
```

**Warning Message in Logs:**

```
📦 WARN: More than one requirements file detected in the repository.
Available options: conda /mount/src/cam-sust2025/environment.yml,
uv /mount/src/cam-sust2025/requirements.txt.
Used: conda with /mount/src/cam-sust2025/environment.yml
```

**Fix**: Delete `environment.yml` and redeploy.

## 🔥 **CRITICAL FIX: Streamlit Configuration Deprecation**

**If you see "global.dataFrameSerialization IS NO LONGER SUPPORTED" or similar config errors:**

**❌ PROBLEM**: `.streamlit/config.toml` contains deprecated configuration options.

**✅ SOLUTION**: Update `.streamlit/config.toml` to remove deprecated options:

**Remove these deprecated settings:**

```toml
# REMOVE - No longer supported
showErrorDetails = true
dataFrameSerialization = "legacy"
```

**Modern Arrow serialization is automatic in Streamlit 1.28+**

---

## 🖥️ **Platform Information**

**Streamlit Community Cloud Environment:**

- **Platform**: `linux-64` (x86_64 architecture)
- **OS**: Debian bookworm (bookworm InRelease)
- **Package Manager**: apt-get (system packages) + conda/pip (Python)
- **Python**: Conda-managed environment with pip fallback
- **Container**: Isolated Linux container per deployment

**This means:**

- ✅ All packages must be Linux-compatible
- ✅ Use Debian/Ubuntu package names in `packages.txt`
- ✅ Binary wheels available for most scientific packages
- ✅ Cross-platform Python packages work (numpy, pandas, etc.)

---

## Common "Error installing requirements" Solutions

### 🎯 **Quick Fix - Use Minimal Requirements**

If you're getting installation errors, try this step-by-step approach:

#### Step 1: Deploy with Minimal Requirements

Use the current `requirements.txt` which has been optimized for Streamlit Cloud with:

- ✅ Fixed versions (not >=)
- ✅ Core packages only
- ✅ Heavy packages commented out

#### Step 2: Verify Basic Deployment

Check that the app loads with:

- Streamlit framework
- Basic scientific computing (numpy, pandas, scipy)
- Matplotlib visualization
- Astropy astronomy core

#### Step 3: Gradually Add Features

If basic deployment works, uncomment packages one by one:

```bash
# Try adding these one at a time:
bagpipes==1.0.0
corner==2.2.1
```

### 🔧 **Common Package Issues**

| Package         | Issue                    | Solution                                         |
| --------------- | ------------------------ | ------------------------------------------------ |
| `bagpipes`      | Compilation errors       | Use specific version `bagpipes==1.0.0` or remove |
| `corner`        | C++ compiler needed      | Remove for now, app has fallback                 |
| `21cmFAST`      | System libraries missing | Keep commented - requires packages.txt           |
| `jwst`          | Very heavy (>1GB)        | Keep commented - timeout issues                  |
| `openai>=1.0.0` | Version conflicts        | Use exact version `openai==1.3.5`                |

### 🐛 **Specific Error Debugging**

#### "Could not install packages due to an EnvironmentError"

**Solution**: Package version conflicts

```bash
# Replace >= with == for specific versions
streamlit==1.28.1
numpy>=2.0.0
pandas==2.0.3
```

#### "Building wheel for [package] failed"

**Solution**: Remove packages that need compilation

```bash
# Comment out these in requirements.txt:
# bagpipes>=1.0.0
# corner>=2.2.0
```

#### "No space left on device"

**Solution**: Too many/too large packages

```bash
# Remove development packages:
# pytest>=7.0.0
# black>=23.0.0

# Remove heavy scientific packages:
# jwst>=1.12.0
```

#### "Timeout during pip install"

**Solution**: Streamlit Cloud has build time limits

```bash
# Keep only essential packages:
streamlit
numpy
pandas
matplotlib
astropy
openai
```

### 🚀 **Recommended Deployment Sequence**

1. **First Deploy**: Use current `requirements.txt` (minimal)
2. **Verify**: App loads and shows status panel
3. **Test Core**: Data upload, basic analysis work
4. **Add Features**: Uncomment packages gradually
5. **Monitor**: Check build logs for any issues

### 📋 **Deployment Checklist**

- ✅ Python version: 3.10
- ✅ Main file: `app.py`
- ✅ Branch: `main`
- ✅ Requirements: Use minimal version first
- ✅ System packages: `packages.txt` present
- ✅ Setup script: `setup.sh` present

### 🔍 **Debug Your Current Error**

To help debug your specific error:

1. **Check Build Logs**: In Streamlit Cloud, click "View logs" during deployment
2. **Find Error Line**: Look for lines starting with "ERROR:" or "Failed"
3. **Identify Package**: Note which package is causing the failure
4. **Apply Fix**: Comment out the problematic package

### 💡 **Pro Tips**

- **Start Simple**: Deploy minimal version first, add complexity later
- **Monitor Logs**: Always check build logs for specific error messages
- **Version Lock**: Use exact versions (==) instead of minimum versions (>=)
- **Fallback Ready**: Your app is designed to work without heavy packages

### 🆘 **Emergency Minimal Requirements**

If nothing works, try this ultra-minimal `requirements.txt`:

```txt
streamlit==1.28.1
numpy>=2.0.0
pandas==2.0.3
matplotlib==3.7.2
```

This should always deploy successfully and give you a working base to build upon.

---

**Need Help?** Check the specific error message in Streamlit Cloud build logs and match it to the solutions above!
