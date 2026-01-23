import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pathlib
import os
import glob

base_path = pathlib.Path(__file__).parent.resolve()

# Choose font
font_name = "Times New Roman"

# Function to get list of font families
def get_families(path_list):
    font_family_list = []
    for fname in path_list:
        try:
            family_name = fm.get_font(fname).family_name
            font_family_list.append(family_name)
        except:
            pass
    return font_family_list

# List system font families 
system_font_paths = fm.findSystemFonts()
system_font_family_list = get_families(system_font_paths)

# List local font families
fonts_dir = str(base_path.joinpath('fonts'))
local_font_paths = glob.glob(str(base_path.joinpath('fonts').joinpath("*").joinpath("*")))
local_font_family_list = get_families(local_font_paths)

# Get font
def get_font(font_name,default_name):

    # Check local directory for font:
    if font_name in local_font_family_list:
        font_folders = glob.glob(str(base_path.joinpath('fonts').joinpath("*")))
        for font_folder in font_folders:
            for font in fm.findSystemFonts(font_folder):
                try:
                    family_name = fm.get_font(font).family_name
                    if family_name == font_name:
                        fm.fontManager.addfont(font)
                except:
                    pass
        set_font_name = font_name
    
    # Check system for font
    elif font_name in system_font_family_list:
        set_font_name = font_name

    else:
        print(f"{font_name} not found. Defaulting to {default_name}")
        set_font_name = default_name

    return set_font_name

# Function to turn on/off usetex and set preamble
def usetex(flag):
    if local_flag&flag:
        plt.rcParams['text.usetex']='True' # tells Matplotlib to use your external latex
        plt.rcParams['text.latex.preamble'] = r'\usepackage{newtxtext}\usepackage{newtxmath}'
        plt.rcParams['font.family'] = 'serif'
    else:
        plt.rcParams['text.usetex']='False'
        plt.rcParams['axes.formatter.use_mathtext'] = True
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rc('axes', unicode_minus=False)

        set_font_name = get_font(font_name,"serif")
        plt.rcParams["font.family"] = set_font_name

# Determine whether to use Mathtext or local TeX installation
home = str(pathlib.Path("~").expanduser())
if home == '/Users/cydavid':
    local_flag = True
    latex_path = '/usr/local/texlive/2022/bin/universal-darwin' # Set this to where latex is located on your computer
    if latex_path in os.environ["PATH"]:
        print('already in PATH')
    else:
        os.environ["PATH"] += os.pathsep + latex_path
        print('added to PATH')

    usetex(True)
else:
    local_flag = False
    usetex(False)

plt.rcParams['hatch.linewidth'] = 0.4