import logging
import traceback

from dependencies import pyplastimatch


def plastimatch_hd(reference_structure_path: str, other_structure_path: str):
    """
    Depends on plastimatch bein installed. Look up documentation on: https://plastimatch.org/getting_started.html

    :param reference_structure_path: Path to reference nifti.
    :param other_structure_path: Path to other nifti.
    :return: dict with difference hausdorff distances.
    """
    try:
        counter = 0
        while counter < 10:
            try:
                d = pyplastimatch.hd(reference_structure_path, other_structure_path)
                return d
            except Exception as e:
                print(e)
                counter += 1
    except Exception as e:
        logging.error(other_structure_path, str(e), traceback.format_exc())
        raise e
