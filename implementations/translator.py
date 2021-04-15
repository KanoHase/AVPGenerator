def text2fasta(filename_list):
  fasta_dir = "fasta_data"

  for p_file in filename_list:
      with open("./data/"+ p_file + ".txt") as f:
          with open("./" + fasta_dir + "/"+ p_file + ".fasta", "w") as g:
              i = 1
              for line in f:
                  tmp = ">" + str(i) + "\n" + line[:-1] +"\n"
                  g.write(tmp)
                  i +=1

text2fasta(["amino_positive_541", "amino_negative_541"])
