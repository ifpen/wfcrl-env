"""
Adapted from openfast_toolbox example `Ex2_FFarmInputSetup.py`
@ https://github.com/OpenFAST/openfast_toolbox/blob/main/openfast_toolbox/fastfarm/examples/
"""
import glob
import os
import shutil

from openfast_toolbox.fastfarm import fastFarmTurbSimExtent, writeFastFarm
from openfast_toolbox.io.fast_input_file import FASTInputFile

TEMPLATE_DIR = "simulators/fastfarm/inputs/template/"
CASE_DIR = "simulators/fastfarm/inputs/"
SERVO_DIR = "simulators/fastfarm/servo_dll/"


def clean_folder(path):
    for subpath in glob.glob(path):
        if os.path.isdir(subpath):
            shutil.rmtree(subpath)
        else:
            os.remove(subpath)


def create_ff_case(xWT, yWT, zWT, max_iter, dt):
    template_dir = TEMPLATE_DIR
    output_dir = CASE_DIR
    servoDir = SERVO_DIR
    clean_folder(f"{output_dir}FarmInputs/*")
    clean_folder(f"{output_dir}5MW_Baseline/*")
    templateFSTF = os.path.join(f"{template_dir}FarmInputs/", "Case.fstf")
    outputFSTF = os.path.join(f"{output_dir}FarmInputs/", "Case.fstf")

    fstf = FASTInputFile(templateFSTF)
    ref_path = fstf["WindTurbines"][0, 3]
    max_time = max_iter * dt

    fst = FASTInputFile(
        os.path.join(f"{template_dir}FarmInputs/", ref_path.replace('"', ""))
    )
    ed = FASTInputFile(
        os.path.join(f"{template_dir}FarmInputs/", fst["EDFile"].replace('"', ""))
    )
    inflow = FASTInputFile(
        os.path.join(f"{template_dir}FarmInputs/", fst["InflowFile"].replace('"', ""))
    )

    # --- Parameters for TurbSim Extent
    # Turbine diameter (m)
    R = ed["TipRad"]
    D = R * 2
    # Hub Height (m)
    hubHeight = ed["TowerHt"]
    # x-extent of high res box in diamter around turbine location
    extent_X_high = 1.2  # np.round((fstf['NX_High'] * dX_High)/D,2)
    # y-extent of high res box in diamter around turbine location
    extent_YZ_high = 1.2  # np.round((fstf['NY_High'] * dYZ_High)/D,2)
    # maximum blade chord (m). Turbine specific.
    chord_max = 5
    # Meandering constant (-)
    Cmeander = 1.9

    # TurbSim Box to be used in FAST.Farm simulation, need to exist.
    BTS_filename = os.path.join(
        f"{template_dir}FarmInputs/", inflow["FileName_BTS"].replace('"', "")
    )

    out_list_sel = None

    # --- Get box extents
    FFTS = fastFarmTurbSimExtent(
        BTS_filename,
        hubHeight,
        D,
        xWT,
        yWT,
        Cmeander=Cmeander,
        chord_max=chord_max,
        extent_X=extent_X_high,
        extent_YZ=extent_YZ_high,
        meanUAtHubHeight=True,
    )

    # --- Write Fast Farm file with layout and Low and High res extent
    writeFastFarm(
        outputFSTF, templateFSTF, xWT, yWT, zWT, FFTS=FFTS, OutListT1=out_list_sel
    )
    print("Created FAST.Farm input file:", outputFSTF)

    servo_file_name = fst["ServoFile"]
    servo = FASTInputFile(
        os.path.join(f"{template_dir}FarmInputs/", servo_file_name.replace('"', ""))
    )
    servo_dll_filename = servo["DLL_FileName"]
    servo_dll_file = os.path.join(servoDir, servo_dll_filename.replace('"', ""))

    # Copy all other files
    shutil.copytree(
        f"{template_dir}5MW_Baseline/", f"{output_dir}5MW_Baseline/", dirs_exist_ok=True
    )
    for file in glob.glob(f"{template_dir}FarmInputs/*.bts"):
        shutil.copy2(file, f"{output_dir}FarmInputs/")
    for file in glob.glob(f"{template_dir}FarmInputs/*.dat"):
        shutil.copy2(file, f"{output_dir}FarmInputs/")
    # Get needed .fst files
    out_fstf = FASTInputFile(outputFSTF)
    fst_files = [row[3].replace('"', "") for row in out_fstf["WindTurbines"]]
    for i, file in enumerate(fst_files):
        fst["ServoFile"] = servo_file_name.replace("1", str(i + 1))
        fst.write(os.path.join(f"{output_dir}FarmInputs/", file))
        servo["DLL_FileName"] = servo_dll_filename.replace("1", str(i + 1))
        servo.write(
            os.path.join(f"{output_dir}FarmInputs/", fst["ServoFile"].replace('"', ""))
        )
        shutil.copy2(
            servo_dll_file,
            os.path.join(
                f"{output_dir}FarmInputs/", servo["DLL_FileName"].replace('"', "")
            ),
        )

    out_fstf["TMax"] = max_time
    out_fstf["DT_Low"] = dt
    out_fstf.write(outputFSTF)
    return outputFSTF
