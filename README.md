# Welcome to the Kthr Lab's music generator!

[![N|Solid](https://www.steinway.com/.imaging/mte/steinway-us-theme/Medium-Rectangle/dam/spirio/landscape/PB_STEINWAY_SON_030215_17_226_CMJN300_fma.jpg/jcr:content/PB_STEINWAY_SON_030215_17_226_CMJN300_fma.jpg)](https://nodesource.com/products/nsolid)

Referals<br>
[The original professor Kitahara's code](https://colab.research.google.com/drive/1dgRg3bPt5AwvbNUDWHiga7Xn3W-0O_ut#scrollTo=8PFf8Dd3Ob_7)
<br>
[TF official documentation for music generation](https://www.tensorflow.org/tutorials/audio/music_generation)<br>
[Tensorflow tutorials on YouTube](https://www.youtube.com/watch?v=gRRGr_tJnAA&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb&index=8)

<br>
The kthrPiano series is the one you want to try.
<br>It has 1-5 versions.
When you see the audio output for 2 seconds and hear nothing, it means the model generated/ predicted rest nodes.
<br>
<br>1:<br> Honestly, I have no idea what I was trying to do in this version. I think I was trying to generate notes with given 2 notes but I was using the model which predicts the alto part with given soprano part, i.e. a complete mess, ha! 
<br>
<br>2:<br>
The model generates the one single note of target Y with given two notes X.
<br> Input (Xn, Xn+1) >>> Output (Xn+2)
<br> Look at the printed out message, if the model generated, you will see<br>
"Got y value!!!!!!! {values}<br>
y pred index!! {note index}"<br>
You can add 36 to the note index to know the exact pitch number the midi plays.
<br> Overall, this code is successfully implemented.
<br>
<br>3:<br>
(Xn, Xn+1) -> Xn+2 and then use Xn+2 to predict Xn+3, 
Xn+3 to predict Xn+4, and so forth. I.e. The model generates the entire tune (length 64) with given 2 notes.<br>
Also I implemented some conditionals for more musical approach in play midi. Read comments in there if you want a equally measured beat or randomly generated beat as you wish. You can simply mute <br>art_balance=random.choices(--) <br>to get a stable beat.<br>
The issue here is the start/ end point of the midi process is not properly calculated when you used random.choice for art_balance even though it sounds musical, actually messed up beats backend. 
<br><br>Overall, the model performs correctly and logic for the music genetration is correct except the midi's start-end point handling with random beats. 

<br>
<br>4:<br>
The updated version of version 3.<br>
The goal is the same but this code becomes more dynamic as you can
select the window size(input feature sequence length) from 1-62. <br>
<br>
Also you can choose the window type.
<br><br>Window1:<br> Shape (495*(64-window_size), window_size, 49)<br>
This makes tens of thousand of samples as it extract the window size of notes every step of the remaining length for the entire chorale directory. Meaning if you want to feed 3 notes,the sample will be 495(songs)* 64(the entire length) - 3(your desired feed)=30,195 samples. <br>
This window gives you more accurate and expected results than window 2 as the model can learn enough.
<br><br>
Window2:<br>
Shape (495, window_size, 49)<br>
It comb through each song to extract the notes of the window size.
This window has a style which the professor requested however the 495 samples(tunes) are too few to train the model and it does not generate expected results.<br>
If there are 10,000 and more samples(tunes), I assume the result would be better.<br>
Like, Shape (16000, window_size, 49) is the ideal shape where 16000 represents number of songs.
<br>
<br>
5:<br>
This model is still on the process, where I tried to use the generated melody and use it as the input feature to predict background harmmonies, i.e. tenor and bass.






