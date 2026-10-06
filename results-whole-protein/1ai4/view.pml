reinitialize
load protein.pdb, protein
load pocket.pdb, pocket
load crystal_ligand.pdb, crystal
load pose_001.pdb, pose_001
load pose_002.pdb, pose_002
load pose_003.pdb, pose_003
load pose_004.pdb, pose_004
load pose_005.pdb, pose_005
load pose_006.pdb, pose_006
load pose_007.pdb, pose_007
load pose_008.pdb, pose_008
load pose_009.pdb, pose_009
load pose_010.pdb, pose_010
load pose_011.pdb, pose_011
load pose_012.pdb, pose_012
load pose_013.pdb, pose_013
load pose_014.pdb, pose_014
load pose_015.pdb, pose_015
load pose_016.pdb, pose_016
load pose_017.pdb, pose_017
load pose_018.pdb, pose_018
load pose_019.pdb, pose_019
load pose_020.pdb, pose_020

hide everything
show cartoon, protein
show sticks, crystal
show sticks, pose_001
color green, crystal
color cyan, pose_001
zoom crystal, 12
disable pose_002
disable pose_003
disable pose_004
disable pose_005
disable pose_006
disable pose_007
disable pose_008
disable pose_009
disable pose_010
disable pose_011
disable pose_012
disable pose_013
disable pose_014
disable pose_015
disable pose_016
disable pose_017
disable pose_018
disable pose_019
disable pose_020

# Enable another pose with, for example:
# enable pose_002
# disable pose_001
