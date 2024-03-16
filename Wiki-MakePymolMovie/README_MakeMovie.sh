
cd WorkingDir/

mkdir DataRepo

for icif in 1*_BoxCox_*_[6-9].cif
do

    fnn=$(echo -e "$icif" | sed 's#.cif##g' )

     
    if [[ -f "./Movie_${fnn}.gif" ]] 
       then
           continue
       else
           echo -e "start ${fnn}"
    fi


    echo -e "$fnn"
   
    sed "s/4tst/"${fnn}"/g" ../Script/Templates/Pymol_Movie.pml > Pymol_Movie_temp.pml

    if [[ $fnn =~ '3j3q' ]] ; then
        echo -e "3j3q has only 4 states " 
        sed -i "s#set state, 5,#set state, 4,#g" Pymol_Movie_temp.pml
        sed -i "s#set state, 6,#set state, 4,#g" Pymol_Movie_temp.pml
    fi


    /home/homingla/Software/PyMOL-2.5.1_283-Linux-x86_64-py37/bin/pymol -cq Pymol_Movie_temp.pml
    #/home/homingla/Software/PyMOL-2.5.1_283-Linux-x86_64-py37/bin/pymol Pymol_Movie_temp.pml

    echo -e "finishing convert to gif"


    cp Movie_${fnn}_06.png Movie_${fnn}_07.png
    cp Movie_${fnn}_05.png Movie_${fnn}_08.png
    cp Movie_${fnn}_04.png Movie_${fnn}_09.png
    cp Movie_${fnn}_03.png Movie_${fnn}_10.png
    cp Movie_${fnn}_02.png Movie_${fnn}_11.png
    cp Movie_${fnn}_01.png Movie_${fnn}_12.png




    convert -delay 2 -loop 0 -dispose 2 Movie_${fnn}*.png Movie_${fnn}.gif

    rm Movie_${fnn}*.png
    mv ${icif} DataRepo/ 

    #pkill -9 pymol
    #pkill -9 python
    #pkill -9 convert
    #exit

done



cd ../
