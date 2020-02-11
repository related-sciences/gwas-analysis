
def hapmap3_hg18_rg():
    """Get Hail reference genome properties for HapMap Phase III data
    
    This is based largely on the NCBI statistics for hg18 at https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.12/#/st_Primary-Assembly (with minor 
    accomodations in non-autosomes)
    """
    return dict(
        name='hapmap3_hg18',
        # Contigs 23 and 25 are not official names yet they appear in the HapMap III data
        contigs=[
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', 
            '15', '16', '17', '18', '19', '20', '21', '22', '23', '25'
        ],
        # hg18 spec ==> X = 154913754, Y = 57772954 but use X length for 23 + 25
        # Note: 23 and 25 both have loci that exceed the length of Y chromosome, so it's unclear what they 
        # actually represent (both have all loci < max X though).  For the tutorial, it makes no difference
        # though as long as hail doesn't drop them.
        lengths={
            '1': 247249719, '2': 242951149, '3': 199501827, '4': 191273063, '5': 180857866, '6': 170899992, '7': 158821424, '8': 146274826, 
            '9': 140273252, '10': 135374737, '11': 134452384, '12': 132349534, '13': 114142980, '14': 106368585, '15': 100338915, '16': 88827254, 
            '17': 78774742, '18': 76117153, '19': 63811651, '20': 62435964, '21': 46944323, '22': 49691432, '23': 154913754, '25': 154913754
        },
        x_contigs='23'
    )

