# groom_clip.npz

Recorded fly foreleg grooming kinematics, from the original NeuroMechFly repository
(https://github.com/NeLy-EPFL/NeuroMechFly, Apache License 2.0, copy in this folder).
File: data/joint_tracking/grooming/fly1/df3d/joint_angles__180921_aDN_CsCh_Fly6_003_SG1_behData_images_images.pkl

Cite: Lobato-Rios V, Tata Ramalingasetty S, Ozdil PG, Arreguit J, Ijspeert AJ, Ramdya P (2022).
NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster. Nature Methods 19, 620-627.
doi:10.1038/s41592-022-01466-7

Changes made: front legs only, 100 Hz subsample, Savitzky-Golay smoothing, one loopable segment cut out.
Rebuild with tools/make_groom_clip.py.
