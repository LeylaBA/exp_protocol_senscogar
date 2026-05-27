# exp_protocol_senscogar
The experimental protocol I developed using PsychoPy, for a published study that I co-first-authored. 

The protocol consists of 2 experimental sessions and a practice session. In both experimental sessions, participants are presented with a set of visual stimuli that they can pass or skip. Which of the experimental session participants will start with, can be set by choosing group 1 or 2 in the beginning of the experiment. So any such sort of experimental design can utilize the same protocol. 

Within every trial there's a "tag" code-component that sends requests to a database. This has been implemented to extract succesful trials, skipped ones and the exact time of each of these events. To avoid errors, edit this part or completely delete it (psychopy uses its own experiment outputs anyway).

Also you need to adjust the paths in experiment_design>conditions and also in the experiment script or interface.
