#!/bin/bash


DownloadFolder="/scratch1/homingla/Project-InchingBenchmarkDownload/"


for i_PdbCif in Pdb Cif CifShowcase
do


    for i_machine in A100
    do

# =====
# mkdir
# =======
        # ==========================
        # Time to GPU preprocess
        # =========================
        mkdir ${DownloadFolder}/BenchmarkLinuxTimePreprocessing0064_${i_machine}

        # ==========================
        # Inching TRLM
        # ==========================
        mkdir ${DownloadFolder}/BenchmarkLinuxInchingTRLM0064_${i_machine}
        mkdir ${DownloadFolder}/BenchmarkLinuxInchingTRLMHD0064_${i_machine}

        # ================================================
        # Inching JDM
        # ================================================
        mkdir ${DownloadFolder}/BenchmarkLinuxInchingJDM0064_${i_machine}
        mkdir ${DownloadFolder}/BenchmarkLinuxInchingJDMHD0064_${i_machine}


        # ================================================
        # Inching IRLM
        # ================================================
        mkdir ${DownloadFolder}/BenchmarkLinuxInchingIRLM0064_${i_machine}
        mkdir ${DownloadFolder}/BenchmarkLinuxInchingIRLMHD0064_${i_machine}

        # ==========================
        # Inching TRLM
        # ==========================
        mkdir ${DownloadFolder}/BenchmarkLinuxInchingTRLMFull0064_${i_machine}
        mkdir ${DownloadFolder}/BenchmarkLinuxInchingTRLMHDFull0064_${i_machine}

        # ================================================
        # Inching JDM
        # ================================================
        mkdir ${DownloadFolder}/BenchmarkLinuxInchingJDMFull0064_${i_machine}
        mkdir ${DownloadFolder}/BenchmarkLinuxInchingJDMHDFull0064_${i_machine}


        # ================================================
        # Inching IRLM
        # ================================================
        mkdir ${DownloadFolder}/BenchmarkLinuxInchingIRLMFull0064_${i_machine}
        mkdir ${DownloadFolder}/BenchmarkLinuxInchingIRLMHDFull0064_${i_machine}


        # ==============================
        # Inching Hessian and ARPACK
        # ===============================
        j_machine='EPYC'
        mkdir ${DownloadFolder}/BenchmarkLinuxArpack0064_${j_machine}

        # ==============================
        # Prody
        # =============================
        mkdir ${DownloadFolder}/BenchmarkLinuxPrody0064_${j_machine}
        mkdir ${DownloadFolder}/BenchmarkLinuxProdySparse0064_${j_machine}


# =================
# Copy
# -================


        # ====================
        # Time to preprocess
        # ====================
        cp ../BenchmarkLinuxTimePreprocessing0064_${i_machine}/*.pkl ${DownloadFolder}/BenchmarkLinuxTimePreprocessing0064_${i_machine}/
        # ==========================
        # Inching TRLM
        # ==========================
        cp ../BenchmarkLinuxInchingTRLM0064_${i_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxInchingTRLM0064_${i_machine}/
        cp ../BenchmarkLinuxInchingTRLMHD0064_${i_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxInchingTRLMHD0064_${i_machine}/




        # ================================================
        # Inching JDM
        # ================================================
        cp ../BenchmarkLinuxInchingJDM0064_${i_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxInchingJDM0064_${i_machine}/
        cp ../BenchmarkLinuxInchingJDMHD0064_${i_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxInchingJDMHD0064_${i_machine}/


        # ================================================
        # Inching IRLM
        # ================================================
        cp ../BenchmarkLinuxInchingIRLM0064_${i_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxInchingIRLM0064_${i_machine}
        cp ../BenchmarkLinuxInchingIRLMHD0064_${i_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxInchingIRLMHD0064_${i_machine}


        # ==========================
        # Inching TRLM
        # ==========================
        cp ../BenchmarkLinuxInchingTRLMFull0064_${i_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxInchingTRLMFull0064_${i_machine}/
        cp ../BenchmarkLinuxInchingTRLMHDFull0064_${i_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxInchingTRLMHDFull0064_${i_machine}/




        # ================================================
        # Inching JDM
        # ================================================
        cp ../BenchmarkLinuxInchingJDMFull0064_${i_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxInchingJDMFull0064_${i_machine}/
        cp ../BenchmarkLinuxInchingJDMHDFull0064_${i_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxInchingJDMHDFull0064_${i_machine}/


        # ================================================
        # Inching IRLM
        # ================================================
        cp ../BenchmarkLinuxInchingIRLMFull0064_${i_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxInchingIRLMFull0064_${i_machine}
        cp ../BenchmarkLinuxInchingIRLMHDFull0064_${i_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxInchingIRLMHDFull0064_${i_machine}



        # ==============================
        # Inching Hessian and ARPACK
        # ===============================
        j_machine='EPYC'
        cp ../BenchmarkLinuxArpack0064_${j_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxArpack0064_${j_machine}/

        # ==============================
        # Prody
        # =============================
        cp ../BenchmarkLinuxPrody0064_${j_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxPrody0064_${j_machine}/
        cp ../BenchmarkLinuxProdySparse0064_${j_machine}/Perf*.pkl ${DownloadFolder}/BenchmarkLinuxProdySparse0064_${j_machine}/

    done
done



