# ligand_receptor.py

def Fact(GLP1, semaglutide, Kd_GLP1=0.17, Kd_semaglutide=0.38):
    """Fraction of GLP1R activated by either ligand (all inputs in nM).

    Lecture 2, slide 14: equilibrium competition for one binding site.
    Either bound ligand fully activates GLP1R; free ligand concentrations
    are approximated by the supplied concentrations (negligible depletion).
    """
    glp1_ratio = GLP1 / Kd_GLP1
    semaglutide_ratio = semaglutide / Kd_semaglutide
    return (glp1_ratio + semaglutide_ratio) / (
        1 + glp1_ratio + semaglutide_ratio
    )


if __name__ == "__main__":
    # Lecture 2, slide 15: example concentrations [nM].
    GLP1 = 0.0150                 # fasting GLP1
    semaglutide = 1.0
    print(f"Fraction of GLP1R active: {Fact(GLP1, semaglutide):.4f}")
