# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common functions to spatialy normalize the data.
"""

# Imports
import os
import re
import sys
import logging
import subprocess
import nibabel
import numpy as np
from pynet.utils import TemporaryDirectory


# Global parameters
logger = logging.getLogger("pynet")


def padd(arr, shape, fill_value=0):
    """ Apply a padding.

    Parameters
    ----------
    arr: array
        the input data.
    shape: list of int
        the desired shape.
    fill_value: int, default 0
        the value used to fill the array.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    orig_shape = arr.shape
    padding = []
    for orig_i, final_i in zip(orig_shape, shape):
        shape_i = final_i - orig_i
        half_shape_i = shape_i // 2
        if shape_i % 2 == 0:
            padding.append((half_shape_i, half_shape_i))
        else:
            padding.append((half_shape_i, half_shape_i + 1))
    for cnt in range(len(arr.shape) - len(padding)):
        padding.append((0, 0))
    return np.pad(arr, padding, mode="constant", constant_values=fill_value)


def downsample(arr, scale):
    """ Apply a downsampling.

    Parameters
    ----------
    arr: array
        the input data.
    scale: int
        the downsampling scale factor in all directions.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    slices = []
    for cnt, orig_i in enumerate(arr.shape):
        if cnt == 3:
            break
        slices.append(slice(0, orig_i, scale))
    return arr[tuple(slices)]


def scale(im, scale, tmpdir=None, check_pkg_version=True):
    """ Scale the MRI image.

    This function is based on FSL.

    Parameters
    ----------
    im: nibabel.Nifti1Image
        the input image.
    scale: int
        the scale factor in all directions.
    tmpdir: str, default None
        a folder where the intermediate results are saved.
    check_pkg_version: boolean, default True
        put to 1 if the package is not installed with the source repository.

    Returns
    -------
    normalized: nibabel.Nifti1Image
        the normalized input image.
    """
    check_version("fsl", check_pkg_version)
    check_command("flirt")
    with TemporaryDirectory(dir=tmpdir, name="scale") as tmpdir:
        input_file = os.path.join(tmpdir, "input.nii.gz")
        trf_file = os.path.join(tmpdir, "trf.txt")
        output_file = os.path.join(tmpdir, "output.nii.gz")
        nibabel.save(im, input_file)
        cmd = ["flirt", "-in", input_file, "-ref", input_file, "-out",
               output_file, "-applyisoxfm", str(scale), "-omat", trf_file]
        logger.debug(" ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        logger.debug("scale : {0} - {1}\n".format(output, err))
        rc = p.returncode
        if rc != 0:
            raise ValueError("\noutput : {0}\n err : {1}".format(output, err))
        normalized = nibabel.load(output_file)
        normalized = nibabel.Nifti1Image(
            normalized.get_data(), normalized.affine)
    return normalized


def bet2(im, frac=0.5, tmpdir=None, check_pkg_version=True):
    """ Skull stripped the MRI image.

    This function is based on FSL.

    Parameters
    ----------
    im: nibabel.Nifti1Image
        the input image.
    tmpdir: str, default None
        a folder where the intermediate results are saved.
    frac: float, default 0.5
        fractional intensity threshold (0->1);smaller values give larger brain
         outline estimates
    check_pkg_version: boolean, default True
        put to 1 if the package is not installed with the source repository.

    Returns
    -------
    skullstripped: nibabel.Nifti1Image
        the skull stripped input image.
    """
    check_version("fsl", check_pkg_version)
    check_command("bet")
    with TemporaryDirectory(dir=tmpdir, name="skullstripped") as tmpdir:
        input_file = os.path.join(tmpdir, "input.nii.gz")
        output_file = os.path.join(tmpdir, "output.nii.gz")
        nibabel.save(im, input_file)
        cmd = ["bet", input_file, output_file, "-f", str(frac), "-B", "-R"]
        logger.debug(" ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        logger.debug("bet2 skullstripped : {0} - {1}\n".format(output, err))
        rc = p.returncode
        if rc != 0:
            raise ValueError("\noutput : {0}\n err : {1}".format(output, err))
        normalized = nibabel.load(output_file)
        normalized = nibabel.Nifti1Image(
            normalized.get_data(), normalized.affine)
    return normalized


def EraseNoise(im, frac=0.2, tmpdir=None, check_pkg_version=True):
    """ Generate binary brain mask and put the noise around the brain to 0.
    Usefull after linear registration.

    This function is based on FSL.

    Parameters
    ----------
    im: nibabel.Nifti1Image
        the input image.
    tmpdir: str, default None
        a folder where the intermediate results are saved.
    frac: float, default 0.2
        fractional intensity threshold (0->1);smaller values give larger brain
         outline estimates
    check_pkg_version: boolean, default True
        put to 1 if the package is not installed with the source repository.

    Returns
    -------
    denoised: nibabel.Nifti1Image
        the denoised input image.
    """
    check_version("fsl", check_pkg_version)
    check_command("bet")
    with TemporaryDirectory(dir=tmpdir, name="denoised") as tmpdir:
        input_file = os.path.join(tmpdir, "input.nii.gz")
        output_file = os.path.join(tmpdir, "output.nii.gz")
        nibabel.save(im, input_file)
        cmd = ["bet", input_file, output_file, "-f", str(frac), "-m"]
        logger.debug(" ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        logger.debug("EraiseNoise : {0} - {1}\n".format(output, err))
        rc = p.returncode
        if rc != 0:
            raise ValueError("\noutput : {0}\n err : {1}".format(output, err))
        denoised = nibabel.load(output_file)
        denoised = nibabel.Nifti1Image(
            denoised.get_data(), denoised.affine)
    return denoised


def super_bet2(im, target, frac=0.5, tmpdir=None, check_pkg_version=True):
    """ Skull stripped the MRI image.
    Performs a much better brain segmenation than the traditional BET2 with
    default configuration. It estimate the inner and outer skull surfaces
    as well as the outer scalp surface, if you have good quality images.
    Inspired by : https://github.com/saslan-7/super-bet2
    4 steps:
        generates brain surface as mesh in .vtk format
        register the T1 to MNI
        using a registered T1 and brain surface, estimates inner and outer
            skull surfaces and outer scalp surface 
        using inner skull surface (mask) as skullstripped method


    This function is based on FSL.

    Parameters
    ----------
    im: nibabel.Nifti1Image
        the input image.
    target: nibabel.Nifti1Image
        the target image.
    tmpdir: str, default None
        a folder where the intermediate results are saved.
    frac: float, default 0.5
        fractional intensity threshold (0->1);smaller values give larger brain
         outline estimates
    check_pkg_version: boolean, default True
        put to 1 if the package is not installed with the source repository.

    Returns
    -------
    skullstripped: nibabel.Nifti1Image
        the skull stripped input image.
    """
    check_version("fsl", check_pkg_version)
    check_command("bet")
    check_command("flirt")
    with TemporaryDirectory(dir=tmpdir, name="brain") as tmpdir:
        input_file = os.path.join(tmpdir, "input.nii.gz")
        target_file = os.path.join(tmpdir, "target.nii.gz")
        brain_file = os.path.join(tmpdir, "brain.nii.gz")
        output_flirt = os.path.join(tmpdir, "flirt_result")
        output_flirtmat = os.path.join(tmpdir, "flirt_result.mat")
        output_mesh_vtk = os.path.join(tmpdir, "brain_mesh.vtk")
        output_mesh_off = os.path.join(tmpdir, "brain_mesh.off")
        nibabel.save(im, input_file)
        nibabel.save(target, target_file)
        brain = os.path.join(tmpdir, "brain")
        cmd = ["bet", input_file, brain, "-f", str(frac), "-e", "-B"]
        logger.debug(" ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        logger.debug("EraiseNoise : {0} - {1}\n".format(output, err))
        rc = p.returncode
        if rc != 0:
            raise ValueError("\noutput : {0}\n err : {1}".format(output, err))

        cmd2 = ["flirt", "-in", brain_file, "-ref", target_file,
                "-out", output_flirt, "-omat", output_flirtmat]
        logger.debug(" ".join(cmd2))
        p = subprocess.Popen(cmd2, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            raise ValueError("\noutput : {0}\n err : {1}".format(output, err))

        cmd3 = ["mv", output_mesh_vtk, output_mesh_off]
        logger.debug(" ".join(cmd3))
        p = subprocess.Popen(cmd3, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            raise ValueError("\noutput : {0}\n err : {1}".format(output, err))

        flirtout = os.path.join("brain", output_flirtmat)
        name = os.path.join(tmpdir, "brain")
        cmd4 = ["betsurf", "-1", "-m", "-s", input_file, output_mesh_off,
                flirtout, name]
        logger.debug(" ".join(cmd4))
        p = subprocess.Popen(cmd4, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            raise ValueError("\noutput : {0}\n err : {1}".format(output, err))
        print("name \n\n\n", tmpdir)
        inskull = os.path.join(tmpdir, "brain_inskull_mask.nii.gz")
        brain = nibabel.load(inskull)
        mask_arr = brain.get_fdata() > 0
        noise_arr = im.get_fdata().squeeze()
        brain_arr = noise_arr*mask_arr
        normalized = nibabel.Nifti1Image(
            brain_arr, im.affine)
    return normalized


def reorient2std(im, tmpdir=None, check_pkg_version=True):
    """ Reorient the MRI image to match the approximate orientation of the
    standard template images (MNI152).

    This function is based on FSL.

    Parameters
    ----------
    im: nibabel.Nifti1Image
        the input image.
    tmpdir: str, default None
        a folder where the intermediate results are saved.
    check_pkg_version: boolean, default True
        put to 1 if the package is not installed with the source repository.

    Returns
    -------
    normalized: nibabel.Nifti1Image
        the normalized input image.
    """
    check_version("fsl", check_pkg_version)
    check_command("fslreorient2std")
    with TemporaryDirectory(dir=tmpdir, name="reorient2std") as tmpdir:
        input_file = os.path.join(tmpdir, "input.nii.gz")
        output_file = os.path.join(tmpdir, "output.nii.gz")
        nibabel.save(im, input_file)
        cmd = ["fslreorient2std", input_file, output_file]
        logger.debug(" ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        logger.debug("reorient2std : {0} - {1}\n".format(output, err))
        rc = p.returncode
        if rc != 0:
            raise ValueError("\noutput : {0}\n err : {1}".format(output, err))

        normalized = nibabel.load(output_file)
        normalized = nibabel.Nifti1Image(
            normalized.get_data(), normalized.affine)
    return normalized


def biasfield(im, mask=None, nb_iterations=50, convergence_threshold=0.001,
              bspline_grid=(1, 1, 1), shrink_factor=1, bspline_order=3,
              histogram_sharpening=(0.15, 0.01, 200), tmpdir=None,
              check_pkg_version=True):
    """ Perform MRI bias field correction using N4 algorithm.

    This function is based on ITK and ANTS.

    Parameters
    ----------
    im: nibabel.Nifti1Image
        the input image.
    mask: nibabel.Nifti1Image, default None
        the brain mask image.
    nb_iterations: int, default 50
        Maximum number of iterations at each level of resolution. Larger
        values will increase execution time, but may lead to better results.
    convergence_threshold: float, default 0.001
        Stopping criterion for the iterative bias estimation. Larger values
        will lead to smaller execution time.
    bspline_grid: int, default (1, 1, 1)
        Resolution of the initial bspline grid defined as a sequence of three
        numbers. The actual resolution will be defined by adding the bspline
        order (default is 3) to the resolution in each dimension specified
        here. For example, 1,1,1 will result in a 4x4x4 grid of control points.
        This parameter may need to be adjusted based on your input image.
        In the multi-resolution N4 framework, the resolution of the bspline
        grid at subsequent iterations will be doubled. The number of
        resolutions is implicitly defined by Number of iterations parameter
        (the size of this list is the number of resolutions).
    shrink_factor: int, default 1
        Defines how much the image should be upsampled before estimating the
        inhomogeneity field. Increase if you want to reduce the execution
        time. 1 corresponds to the original resolution. Larger values will
        significantly reduce the computation time.
    bspline_order: int, default 3
        Order of B-spline used in the approximation. Larger values will lead
        to longer execution times, may result in overfitting and poor result.
    histogram_sharpening: 3-uplate, default (0.15, 0.01, 200)
        A vector of up to three values. Non-zero values correspond to Bias
        Field Full Width at Half Maximum, Wiener filter noise, and Number of
        histogram bins.
    tmpdir: str, default None
        a folder where the intermediate results are saved.
    check_pkg_version: boolean, default True
        put to 1 if the package is not installed with the source repository.

    Returns
    -------
    normalized: nibabel.Nifti1Image
        the normalized input image.
    """
    check_version("ants", check_pkg_version)
    check_command("N4BiasFieldCorrection")
    with TemporaryDirectory(dir=tmpdir, name="biasfield") as tmpdir:
        input_file = os.path.join(tmpdir, "input.nii.gz")
        mask_file = os.path.join(tmpdir, "mask.nii.gz")
        output_file = os.path.join(tmpdir, "output.nii.gz")
        biasfield_file = os.path.join(tmpdir, "biasfield.nii.gz")
        nibabel.save(im, input_file)
        ndim = im.ndim
        bspline_grid = [str(e) for e in bspline_grid]
        histogram_sharpening = [str(e) for e in histogram_sharpening]
        cmd = [
            "N4BiasFieldCorrection",
            "-d", str(ndim),
            "-i", input_file,
            "-s", str(shrink_factor),
            "-b", "[{0}, {1}]".format("x".join(bspline_grid), bspline_order),
            "-c", "[{0}, {1}]".format(
                "x".join([str(nb_iterations)] * 4), convergence_threshold),
            "-t", "[{0}]".format(", ".join(histogram_sharpening)),
            "-o", "[{0}, {1}]".format(output_file, biasfield_file),
            "-v"]
        if mask is not None:
            nibabel.save(mask, mask_file)
            cmd += ["-x", mask_file]
        logger.debug(" ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        logger.debug("biasfield : {0} - {1}\n".format(output, err))
        rc = p.returncode
        if rc != 0:
            raise ValueError("\noutput : {0}\n err : {1}".format(output, err))
        normalized = nibabel.load(output_file)
        normalized = nibabel.Nifti1Image(
            normalized.get_data(), normalized.affine)
    return normalized


def register(im, target, mask=None, cost="normmi", bins=256,
             interp="spline", dof=9, tmpdir=None, check_pkg_version=True):
    """ Register the MRI image to a target image using an affine transform
    with 9 dofs.

    This function is based on FSL.

    Parameters
    ----------
    im: nibabel.Nifti1Image
        the input image.
    target: nibabel.Nifti1Image
        the target image.
    mask: nibabel.Nifti1Image, default None
        the white matter mask image needed by the bbr cost function.
    cost: str, default 'normmi'
        Choose the most appropriate metric: 'mutualinfo', 'corratio',
        'normcorr', 'normmi', 'leastsq', 'labeldiff', 'bbr'.
    bins: int, default 256
        Number of histogram bins
    interp: str, default 'spline'
        Choose the most appropriate interpolation method: 'trilinear',
        'nearestneighbour', 'sinc', 'spline'.
    dof: int, default 9
        Number of affine transform dofs.
    tmpdir: str, default None
        a folder where the intermediate results are saved.
    check_pkg_version: boolean, default True
        put to 1 if the package is not installed with the source repository.

    Returns
    -------
    normalized: nibabel.Nifti1Image
        the normalized input image.
    """
    check_version("fsl", check_pkg_version)
    check_command("flirt")
    with TemporaryDirectory(dir=tmpdir, name="register") as tmpdir:
        input_file = os.path.join(tmpdir, "input.nii.gz")
        target_file = os.path.join(tmpdir, "target.nii.gz")
        trf_file = os.path.join(tmpdir, "trf.txt")
        output_file = os.path.join(tmpdir, "output.nii.gz")
        nibabel.save(im, input_file)
        nibabel.save(target, target_file)
        cmd = ["flirt",
               "-in", input_file,
               "-ref", target_file,
               "-cost", cost,
               "-searchcost", cost,
               "-anglerep", "euler",
               "-bins", str(bins),
               "-interp", interp,
               "-dof", str(dof),
               "-out", output_file,
               "-omat", trf_file,
               "-verbose", "1"]
        if cost == "bbr":
            if mask is not None:
                nibabel.save(mask, mask_file)
            else:
                raise ValueError("A white matter mask image is needed by the "
                                 "bbr cost function.")
            cmd += ["-wmseg", mask_file]
        logger.debug(" ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        logger.debug("register : {0} - {1}\n".format(output, err))
        rc = p.returncode
        if rc != 0:
            raise ValueError("\noutput : {0}\n err : {1}".format(output, err))
        normalized = nibabel.load(output_file)
        normalized = nibabel.Nifti1Image(
            normalized.get_data(), normalized.affine)
    return normalized


def brainmask(im, mask, tmpdir=None, check_pkg_version=True):
    """ Apply cat12vbm brain mask.

    This function is based on cat12vbm masks.

    Parameters
    ----------
    im: nibabel.Nifti1Image
        the input image.
    mask: nibabel.Nifti1Image, default None
        the partial volume p0 label mask image.
    tmpdir: str, default None
        a folder where the intermediate results are saved.
    check_pkg_version: boolean, default True
        put to 1 if the package is not installed with the source repository.

    Returns
    -------
    brain extracted image : nibabel.Nifti1Image
        the skullstripped input image.
    """
    with TemporaryDirectory(dir=tmpdir, name="skullstrippedcat12") as tmpdir:
        input_file = os.path.join(tmpdir, "input.nii.gz")
        mask_file = os.path.join(tmpdir, "mask.nii.gz")
        output_file = os.path.join(tmpdir, "output.nii.gz")
        nibabel.save(im, input_file)
        nibabel.save(mask, mask_file)

        maskPV = mask.get_fdata() > 0
        noise = im.get_fdata().squeeze()
        skullstripped = noise*maskPV

        normalized = nibabel.Nifti1Image(
            skullstripped, im.affine)
        nibabel.save(normalized, output_file)
    return normalized


def apply(im, target, affines, interp="spline", tmpdir=None,
          check_pkg_version=True):
    """ Apply affine transformations to an image.

    This function is based on FSL.

    Parameters
    ----------
    im: nibabel.Nifti1Image
        the input image.
    target: nibabel.Nifti1Image
        the target image.
    affines: str or list of str
        the affine transforms to be applied. If multiple transforms are
        specified, they are first composed.
    interp: str, default 'spline'
        Choose the most appropriate interpolation method: 'trilinear',
        'nearestneighbour', 'sinc', 'spline'.
    tmpdir: str, default None
        a folder where the intermediate results are saved.
    check_pkg_version: boolean, default True
        put to 1 if the package is not installed with the source repository.

    Returns
    -------
    normalized: nibabel.Nifti1Image
        the normalized input image.
    """
    check_version("fsl", check_pkg_version)
    check_command("flirt")
    if not isinstance(affines, list):
        trf_file = affines
    elif len(affines) == 0:
        raise ValueError("No transform specified.")
    else:
        trf_file = os.path.join(tmpdir, "trf.txt")
        affines = [np.loadtxt(path) for path in affines][::-1]
        affine = affines[0]
        for matrix in affines[1:]:
            affine = np.dot(matrix, affine)
        np.savetxt(trf_file, affine)
    with TemporaryDirectory(dir=tmpdir, name="apply") as tmpdir:
        input_file = os.path.join(tmpdir, "input.nii.gz")
        target_file = os.path.join(tmpdir, "target.nii.gz")
        output_file = os.path.join(tmpdir, "output.nii.gz")
        nibabel.save(im, input_file)
        nibabel.save(target, target_file)
        cmd = ["flirt",
               "-in", input_file,
               "-ref", target_file,
               "-init", trf_file,
               "-interp", interp,
               "-applyxfm",
               "-out", output_file]
        logger.debug(" ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            raise ValueError("\noutput : {0}\n err : {1}".format(output, err))
        normalized = nibabel.load(output_file)
        normalized = nibabel.Nifti1Image(
            normalized.get_data(), normalized.affine)
    return normalized


def check_command(command):
    """ Check if a command is installed.

    This function is based on which.

    Parameters
    ----------
    command: str
        the name of the command to locate.
    """
    if sys.platform != "linux":
        raise ValueError("This code works only on a linux machine.")
    process = subprocess.Popen(
        ["which", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode("utf8")
    stderr = stderr.decode("utf8")
    exitcode = process.returncode
    if exitcode != 0:
        logger.debug("Command {0}: {1}".format(command, stderr))
        raise ValueError("Impossible to locate command '{0}'.".format(command))


def check_version(package_name, check_pkg_version):
    """ Check installed version of a package.

    This function is based on dpkg.

    Parameters
    ----------
    package_name: str
        the name of the package we want to check the version.
    """
    process = subprocess.Popen(
        ["dpkg", "-s", package_name],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode("utf8")
    stderr = stderr.decode("utf8")
    exitcode = process.returncode

    if check_pkg_version:
        # local computer installation
        if exitcode != 0:
            version = None
            logger.debug("Version {0}: {1}".format(package_name, stderr))
            raise ValueError(
                "Impossible to check package '{0}' version."
                .format(package_name))
        else:
            versions = re.findall("Version: .*$", stdout, re.MULTILINE)
            version = "|".join(versions)
    else:
        # specific installation
        version = "custom install (no check)."
    logger.info("{0} - {1}".format(package_name, version))
