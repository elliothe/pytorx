'''This script is used to generate the netlist of
the crossbar that can be used in a spice based simulator.
rout{i} is the load resistance of the ADC.
'''
# parameters
vdd = 3.3
vin = vdd
gmax = 1/3e3
gmin = 1/3e6
gloaddac = 1
gloadadc = 2
gwire = 1
crxb_size = 16
gmat = 0.5
# g = np.random.random((crxb_size, crxb_size))
filename = "crxb.sp"
with open(filename, 'w') as f:
    # writing descriptions
    f.write("** Generate for: hspice\n** crosbar size is {0}\n** vdd is {1}\n".format(crxb_size, vdd))

    # writing parameters
    f.write(".OP\n\n")

    # writing input nodes
    for i in range(crxb_size):
        f.write("v{0} input{0} 0 DC={1}\n".format(i, vin))
        f.write("rin{0} input{0} top{0}p0 ".format(i) + str(1/gloaddac) + "\n")

    # writing normal nodes
    for i in range(crxb_size):
        for j in range(crxb_size):
            if i > 0 and j > 0:
                f.write("rrow{0}p{1} top{0}p{2} top{0}p{1} ".format(i, j, j - 1) + str(1 / gwire) + "\n")
                f.write("rcol{0}p{1} bot{2}p{1} bot{0}p{1} ".format(i, j, i - 1) + str(1 / gwire) + "\n")
                f.write("r{0}p{1} top{0}p{1} bot{0}p{1} ".format(i, j) + str(1 / gmat) + "\n")
            elif i == 0 and j > 0:
                f.write("rrow{0}p{1} top{0}p{2} top{0}p{1} ".format(i, j, j - 1) + str(1 / gwire) + "\n")
                f.write("r{0}p{1} top{0}p{1} bot{0}p{1} ".format(i, j) + str(1 / gmat) + "\n")
            elif i > 0 and j == 0:
                f.write("rcol{0}p{1} bot{2}p{1} bot{0}p{1} ".format(i, j, i - 1) + str(1 / gwire) + "\n")
                f.write("r{0}p{1} top{0}p{1} bot{0}p{1} ".format(i, j) + str(1 / gmat) + "\n")
            else:
                f.write("r{0}p{1} top{0}p{1} bot{0}p{1} ".format(i, j) + str(1 / gmat) + "\n")

    # writing output nodes
    for i in range(crxb_size):
        f.write("rout{1} 0 bot{0}p{1} ".format(crxb_size-1, i) + str(1 / gloadadc) + "\n")
    f.write("\n")

    # printing results
    f.write(".print OP ")
    for i in range(crxb_size):

        f.write("i(rout{0}) ".format(i))

    f.write("\n.end")




