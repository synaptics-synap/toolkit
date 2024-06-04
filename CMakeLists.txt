
file(GLOB toolkit *.py *.sh)
list(REMOVE_ITEM toolkit synap_profile.py)
install(PROGRAMS ${toolkit} DESTINATION toolkit/)
file(GLOB toolkit pysynap/*.py pysynap/requirements.txt)
install(FILES ${toolkit} DESTINATION toolkit/pysynap/)
# Dockerfile for synap toolkit
install(FILES ../dockerfiles/Dockerfile_synap DESTINATION dockerfiles)
