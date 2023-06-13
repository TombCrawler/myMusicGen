# Welcome to my very first music generator!

[![N|Solid](https://www.steinway.com/.imaging/mte/steinway-us-theme/Medium-Rectangle/dam/spirio/landscape/PB_STEINWAY_SON_030215_17_226_CMJN300_fma.jpg/jcr:content/PB_STEINWAY_SON_030215_17_226_CMJN300_fma.jpg)](https://nodesource.com/products/nsolid)

Referred to this
https://www.tensorflow.org/tutorials/audio/music_generation
<br>
Not completed yet but for now, just simply run `python3 synthAi.py`
It generates you a piano roll and an example MIDI file. 
<br>
Experiment
| Experiment | Epochs| Intelligence | Musicality| Result                                                                           | Fix                                              |
| ------- | ----------------------- | ----------------- | ---------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------- |
| 1    | 10  | dumb  | slow | Playing each note very long  | Increased epochs. |
| 2     | 50  | arrogant | super fast  | Playing a super fast and random arpeggio | Increased epochs|
| 3    | 100| mad |  lost | stopped at 50 epochs, hit only one chord |decreased epochs | 
| 4   | 30| clever | insane| keep playing higher notes intensely |TBD | 