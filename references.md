# References

Sources used for the algorithms, libraries and standards in this project. Please verify page numbers
and editions before citing them in formal work.

## Robotics fundamentals

1. P. Corke, *Robotics, Vision and Control: Fundamental Algorithms in Python*, 3rd ed., Springer, 2023.
   (Kinematics, Jacobians, trajectories; background for `modelling.py` and `kinematics.py`.)
2. B. Siciliano, L. Sciavicco, L. Villani, G. Oriolo, *Robotics: Modelling, Planning and Control*,
   Springer, 2009.
3. J. J. Craig, *Introduction to Robotics: Mechanics and Control*, 4th ed., Pearson, 2018.
4. J. Denavit and R. S. Hartenberg, "A kinematic notation for lower-pair mechanisms based on
   matrices," *Journal of Applied Mechanics*, 22(2):215-221, 1955. (DH convention used in `RV2FR_DH`.)

## Inverse kinematics and singularities

5. Y. Nakamura and H. Hanafusa, "Inverse kinematic solutions with singularity robustness for robot
   manipulator control," *Journal of Dynamic Systems, Measurement, and Control*, 108(3):163-171, 1986.
   (Adaptive damping in `dls_solve`.)
6. C. W. Wampler, "Manipulator inverse kinematic solutions based on vector formulations and
   damped-least-squares methods," *IEEE Transactions on Systems, Man, and Cybernetics*, 16(1):93-101, 1986.
7. S. R. Buss, "Introduction to inverse kinematics with Jacobian transpose, pseudoinverse and damped
   least squares methods," technical report, University of California San Diego, 2004.
8. T. Yoshikawa, "Manipulability of robotic mechanisms," *The International Journal of Robotics
   Research*, 4(2):3-9, 1985. (`manipulability` in `kinematics.py`.)

## Trajectories

9. T. Flash and N. Hogan, "The coordination of arm movements: an experimentally confirmed
   mathematical model," *Journal of Neuroscience*, 5(7):1688-1703, 1985. (Minimum-jerk / quintic profile.)

## Motion planning and collision checking

10. J. J. Kuffner and S. M. LaValle, "RRT-Connect: An efficient approach to single-query path
    planning," *Proc. IEEE International Conference on Robotics and Automation (ICRA)*, pp. 995-1001, 2000.
11. S. M. LaValle, *Planning Algorithms*, Cambridge University Press, 2006.
12. R. Geraerts and M. H. Overmars, "Creating high-quality paths for motion planning," *The
    International Journal of Robotics Research*, 26(8):845-863, 2007. (Randomised path shortcutting.)
13. C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2005. (Capsule-based collision model.)
14. I. Quilez, "Distance functions" (signed distance to box, sphere and cylinder primitives).
    https://iquilezles.org/articles/distfunctions/

## Computer vision

15. F. Zhang et al., "MediaPipe Hands: On-device real-time hand tracking," arXiv:2006.10214, 2020.
16. C. Lugaresi et al., "MediaPipe: A framework for building perception pipelines," arXiv:1906.08172, 2019.
17. MediaPipe Hand Landmarker documentation. https://developers.google.com/mediapipe
18. G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000. https://opencv.org

## Software libraries

19. P. Corke and J. Haviland, "Not your grandmother's toolbox - the Robotics Toolbox reinvented for
    Python," *Proc. IEEE ICRA*, 2021. https://github.com/petercorke/robotics-toolbox-python
20. J. Haviland, Swift: a lightweight browser-based robotics simulator. https://github.com/jhavl/swift
21. P. Corke, Spatial Maths for Python. https://github.com/petercorke/spatialmath-python
22. J. Haviland, spatialgeometry. https://github.com/jhavl/spatialgeometry
23. C. R. Harris et al., "Array programming with NumPy," *Nature*, 585:357-362, 2020.
24. FastAPI. https://fastapi.tiangolo.com
25. Uvicorn. https://uvicorn.org
26. pytest. https://docs.pytest.org
27. `ir-support-extra-robots` package (visual robot models). Check the package's own repository
    for its source and license.

## Standards and hardware documentation

28. ISO 10218-1 and ISO 10218-2, *Robotics - Safety requirements for industrial robots* (robot and
    robot system / integration). Background only; this project is not certified to any standard.
29. ISO/TS 15066, *Robots and robotic devices - Collaborative robots*, 2016.
30. Mitsubishi Electric, MELFA RV-FR series industrial robot standard specifications / data sheet.
    The nominal DH parameters and joint limits in `modelling.py` must be checked against this document.