import numpy
import itertools
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats
from Bio.Alphabet import IUPAC
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import os
import ntpath
import pickle
import sys
import gc

class ann_result:

    infile=''
    html_table=''
    g_total_fasta=''
    g_all_fasta=''
    g_test_stat=''
    g_mean_arr=''
    g_std_arr=''
    g_table_format={
            "Major capsid": "{:.2f}",
            "Minor capsid": "{:.2f}",
            "Baseplate": "{:.2f}",
            "Major tail": "{:.2f}",
            "Minor tail": "{:.2f}",
            "Portal": "{:.2f}",
            "Tail fiber": "{:.2f}",
            "Tail shaft": "{:.2f}",
            "Collar": "{:.2f}",
            "HTJ": "{:.2f}",
            "Other": "{:.2f}",
            "Confidence": "{:.2%}"
            }


    def __init__(self, filename):
        self.infile=filename
        total_fasta=0
        all_fasta=0
        for record in SeqIO.parse(self.infile, "fasta"):
            all_fasta+=1
            if self.prot_check(str(record.seq)):
                total_fasta+=1
        self.g_total_fasta=total_fasta
        self.g_all_fasta=all_fasta
        self.g_test_stat=pd.read_csv('test_set_stats.csv',index_col=0)
        self.g_mean_arr=pickle.load(open("mean_part.p", "rb" ))
        self.g_std_arr=pickle.load( open("std_part.p", "rb" ))


    def prot_check(self, sequence):
        return set(sequence.upper()).issubset("ABCDEFGHIJKLMNPQRSTVWXYZ*")

    def extract(self):
        AA=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
        SC=["1","2","3","4","5","6","7"]
        tri_pep = [''.join(i) for i in itertools.product(AA, repeat = 3)]
        myseq="AILMVNQSTGPCHKRDEFWY"
        trantab2=myseq.maketrans("AILMVNQSTGPCHKRDEFWY","11111222233455566777")
        tetra_sc = [''.join(i) for i in itertools.product(SC, repeat = 4)]
        total_fasta=self.g_total_fasta
        sec_code=0
        record_current=0
        arr = numpy.empty((total_fasta,10409), dtype=numpy.float)
        names = numpy.empty((total_fasta,1),  dtype=object)
        names_dic=dict()
        for record in SeqIO.parse(self.infile, "fasta"):
            data=(record_current/total_fasta) * 100
            print('extracting features of seq ' + str(record_current+1) + ' of ' + str(total_fasta),end='\r')
            record_current += 1
            ll=len(record.seq)
            seq_name=''
## TODO need to add more checks and better error messages
            if not self.prot_check(str(record.seq)):
                print("Warning: " + record.id + " is not a valid protein sequence")
                continue
            if record.id in names_dic:
                seq_name= record.id + '_' + str(names_dic[record.id])
                names_dic[record.id]=names_dic[record.id]+1
            else:
                seq_name= record.id
                names_dic[record.id]=1
            seqq=record.seq.__str__().upper()
            seqqq=seqq.replace('X','A').replace('J','L').replace('*','A').replace('Z','E').replace('B','D')
           # X = ProteinAnalysis(record.seq.__str__().upper().replace('X','A').replace('J','L').replace('*',''))
            X = ProteinAnalysis(seqqq)
            myseq=seqq.translate(trantab2)
            tt= [X.isoelectric_point(), X.instability_index(),ll,X.aromaticity(),
                 X.molar_extinction_coefficient()[0],X.molar_extinction_coefficient()[1],
                 X.gravy(),X.molecular_weight()]
            tt_n = numpy.asarray(tt,dtype=numpy.float)

            tri_pep_count=[seqq.count(i)/(ll-2) for i in tri_pep]
            tri_pep_count_n = numpy.asarray(tri_pep_count,dtype=numpy.float)
            
            tetra_sc_count=[myseq.count(i)/(ll-3) for i in tetra_sc]
            tetra_sc_count_n = numpy.asarray(tetra_sc_count,dtype=numpy.float)
    
            cat_n= numpy.concatenate((tetra_sc_count_n,tri_pep_count_n,tt_n))
            cat_n = cat_n.reshape((1,cat_n.shape[0]))


            arr[sec_code,:]=cat_n
            names[sec_code,0]=seq_name
            sec_code += 1
        print("\nDone")
        return (names,arr)


    def extract_n(self):
#       normalize
        (names,arr)=self.extract()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if self.g_std_arr[j]==0:
                    pass
                else:
                    arr[i,j]=(arr[i,j] - self.g_mean_arr[j])/self.g_std_arr[j]
        return (names,arr) 

    def predict(self):
        (names,pp)=test.extract_n()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        mean_arr=pickle.load(open("mean_part.p", "rb" ))
        std_arr=pickle.load(open( "std_part.p", "rb" ))
        total_fasta=self.g_total_fasta
        sec_num=0
        arr = numpy.empty((total_fasta,11), dtype=numpy.float)
        n_members = 10
        models = list()
        for model_number in range(n_members):
        #load model
            print('loading ... tetra_sc_tri_p_{:02d}.h5'.format(model_number))
            model =  load_model('tetra_sc_tri_p_'+"{:02d}".format(model_number)+'.h5')
            arr[sec_num,]=model.predict(pp)
            del model
            gc.collect()
            sec_num=sec_num+1
        return (names,arr)


if __name__ == "__main__":
#    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    test=ann_result(sys.argv[1])
    (names,pp)=test.predict()
    print(pp)
