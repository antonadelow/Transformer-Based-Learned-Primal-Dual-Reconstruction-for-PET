# Transformer-Based-Learned-Primal-Dual-Reconstruction-for-PET

Code used in the Master Thesis Transformer-Based Learned
Primal-Dual Reconstruction for PET.

Positron Emission Tomography (PET) is a medical imaging technique that
involves injecting a short-lived radioactive tracer, chemically bound to a
biologically active molecule, into a subject in order to measure cellular
metabolic activity within bodily tissues. By revealing abnormal metabolic
activity, PET serves as an important method in cancer diagnosis. In a realistic
setting, there are multiple sources of noise affecting the measurements. The
Learned Primal-Dual (LPD) reconstruction algorithm, proposed by Adler and
Öktem, utilizes Convolutional Neural Networks (CNNs) in the unrolling to
achieve state of the art results for image reconstruction. The CNNs in the
LPD algorithm impose a locality assumption on features in both the image and
scanner data, which could potentially lead to inaccuracies. The Transformer
architecture could offer advantages over CNNs for this particular problem,
due to its ability to capture global dependencies. Three Transformer-based
architectures were incorporated into the LPD algorithm, compared against
a baseline model, and evaluated on synthetic and experimental data from
a preclinical system. The results show promise in Transformer-based LPD
algorithms, which can provide better reconstructions than previously proposed
CNN-based methods, based on three different figures of merit. Additionally,
a synthetic data generation process designed to mimic a preclinical system is
introduced. The results indicate effective transfer learning from synthetic to
preclinical data.
