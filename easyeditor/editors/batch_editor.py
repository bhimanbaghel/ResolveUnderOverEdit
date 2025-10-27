from enum import Enum


class BatchEditor(Enum):
    CALINET = 'CALINET'
    SERAC = 'SERAC'
    KE = 'KE'
    MEND = 'MEND'
    MEMIT = 'MEMIT'
    MEMIT_RECURSIVE = 'MEMIT_RECURSIVE'
    MEMIT_RECURSIVE_SPREAD = 'MEMIT_RECURSIVE_SPREAD'
    MEMIT_RECURSIVE_NEIGHBOR = 'MEMIT_RECURSIVE_NEIGHBOR'
    PMET = 'PMET'
    PMET_RECURSIVE = 'PMET_RECURSIVE'
    PMET_RECURSIVE_NEIGHBOR = 'PMET_RECURSIVE_NEIGHBOR'
    FT = 'FT'
    QLoRA = 'QLoRA'
    LoRA = 'LoRA'
    EMMET = "EMMET"
    ROME_RECURSIVE = 'ROME_RECURSIVE'
    AlphaEdit_RECURSIVE = 'AlphaEdit_RECURSIVE'
    AlphaEdit_RECURSIVE_NEIGHBOR = 'AlphaEdit_RECURSIVE_NEIGHBOR'

    @staticmethod
    def is_batchable_method(alg_name: str):
        return alg_name == BatchEditor.CALINET.value \
            or alg_name == BatchEditor.SERAC.value \
            or alg_name == BatchEditor.KE.value \
            or alg_name == BatchEditor.MEND.value \
            or alg_name == BatchEditor.MEMIT.value \
            or alg_name == BatchEditor.MEMIT_RECURSIVE.value \
            or alg_name == BatchEditor.MEMIT_RECURSIVE_SPREAD.value \
            or alg_name == BatchEditor.MEMIT_RECURSIVE_NEIGHBOR.value \
            or alg_name == BatchEditor.PMET.value \
            or alg_name == BatchEditor.PMET_RECURSIVE.value \
            or alg_name == BatchEditor.PMET_RECURSIVE_NEIGHBOR.value \
            or alg_name == BatchEditor.FT.value \
            or alg_name == BatchEditor.QLoRA.value \
            or alg_name == BatchEditor.LoRA.value \
            or alg_name == BatchEditor.ROME_RECURSIVE.value \
            or alg_name == BatchEditor.EMMET.value \
            or alg_name == BatchEditor.AlphaEdit_RECURSIVE.value \
            or alg_name == BatchEditor.AlphaEdit_RECURSIVE_NEIGHBOR.value
