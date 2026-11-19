BACKENDS = [
    {"id": "street-gaussians", "label": "Street Gaussians (DriveStudio+)", "runtime": "isolated CUDA (own venv)", "status": "experimental; fflab train --backend drivestudio --method configs/streetgs_ut.yaml"},
    {"id": "omnire", "label": "OmniRe (DriveStudio+)", "runtime": "isolated CUDA (own venv)", "status": "experimental; fflab train --backend drivestudio --method configs/omnire.yaml"},
    {"id": "3dgut", "label": "3DGUT", "runtime": "local-cuda", "status": "planned"},
    {"id": "splatad", "label": "SplatAD", "runtime": "colab-t4", "status": "research"},
    {"id": "idsplat", "label": "IDSplat", "runtime": "isolated", "status": "research"},
    {"id": "storm", "label": "GaussianSTORM", "runtime": "isolated", "status": "research"},
    {"id": "dggt", "label": "DGGT", "runtime": "isolated", "status": "research"},
    {"id": "instant-nurec", "label": "Instant NuRec", "runtime": "isolated CUDA / cloud notebook", "status": "inference runner; local assets required; GPU run unverified"},
    {"id": "unigaussian", "label": "UniGaussian", "runtime": "research", "status": "paper-only"},
    {"id": "gofish", "label": "GOFish", "runtime": "isolated", "status": "research"},
    {"id": "directfisheye-gs", "label": "DirectFisheye-GS", "runtime": "isolated", "status": "research"},
    {"id": "fisheye-gs", "label": "Fisheye-GS", "runtime": "isolated", "status": "research"},
]
