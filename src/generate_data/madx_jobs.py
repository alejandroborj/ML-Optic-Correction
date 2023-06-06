#%%
import cpymad.madx
#import pymadng

#class madx_ml_op(pymadng.MAD):

class madx_ml_op(cpymad.madx.Madx):
    '''Normal cpymad wrapper with extra methods for this exact project'''

    def job_nominal2023(self):
        print(self)
        self.input('''
        option, -echo;
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII_ats.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2022.macros.madx";
        option, echo;

        title, "Model creator for java";

        !@require lhc_runIII_2022.macros.madx

        option, -echo;

        call, file = "/afs/cern.ch/eng/acc-models/lhc/2022/lhc.seq";

        exec, define_nominal_beams();
        call, file="/afs/cern.ch/eng/acc-models/lhc/2022/operation/optics/R2023a_A30cmC30cmA10mL200cm.madx";
        !call, file="/afs/cern.ch/eng/acc-models/lhc/2022/operation/optics/R2023a_A45cmC45cmA10mL200cm.madx";
        exec, cycle_sequences();

        ! BEAM 1

        use, period = LHCB1;

        option, echo;

        exec, match_tunes(62.31, 60.32, 1);

        exec, do_twiss_monitors(LHCB1, "b1_nominal_monitors.dat", 0.0);
        exec, do_twiss_elements(LHCB1, "b1_nominal_elements.dat", 0.0);

        ! BEAM 2

        use, period = LHCB2;

        option, echo;

        exec, match_tunes(62.31, 60.32, 2);
        exec, do_twiss_monitors(LHCB2, "b2_nominal_monitors.dat", 0.0);
        exec, do_twiss_elements(LHCB2, "b2_nominal_elements.dat", 0.0);
        ''')

    def job_magneterrors_b1(self, OPTICS, index, seed):
        self.input('''
        option, -echo;
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runII_ats.macros.madx";
        call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2022.macros.madx";
        call, file = "/afs/cern.ch/eng/lhc/optics/V6.5/errors/Esubroutines.madx"; 
        ! ADDED

        option, echo;

        !@require lhc_runIII_2022.macros.madx

        option, -echo;

        call, file = "/afs/cern.ch/eng/acc-models/lhc/2022/lhc.seq";

        exec, define_nominal_beams();
        !call, file="/afs/cern.ch/eng/acc-models/lhc/2022/operation/optics/R2023a_A30cmC30cmA10mL200cm.madx";
        call, file = "%(OPTICS)s";
        
        option, echo;
        !call, file = "global_corrections.madx";
        option, -echo;
        exec, cycle_sequences();

        ! BEAM 1

        option, echo;
        use, period = LHCB1;
        
        exec, match_tunes(62.31, 60.32, 1);

        ! Assign errors per magnets family:
        ! the same systematic error in each magnet in one family (B2S)
        ! + random component (B2R)
        ! B2R are estimated from WISE
        eoption, seed = %(SEED)s, add=true;
        ON_B2R = 1;
        GCUTR = 3; ! Cut for truncated gaussians (sigmas)

        ! Arc magnets
        select, flag=error, clear;
        select, flag=error, pattern = "^MQ\..*B1";
        Rr = 0.017;
        B2r = 19; ! increased from 18 by 1 unit to reflect MS misalignments
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQM[LC]\..*B1";
        Rr = 0.017;
        B2r = 12;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQM\..*B1";
        Rr = 0.017;
        B2r = 12;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQY\..*B1";
        Rr = 0.017;
        B2r = 11;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQW[AB]\..*B1";
        Rr = 0.017;
        B2r = 15;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQW\..*B1";
        Rr = 0.017;
        B2r = 15;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQT\..*B1";
        Rr = 0.017;
        B2r = 15;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQTL[IH]\..*B1";
        Rr = 0.017;
        B2r = 75;
        exec, SetEfcomp_Q;

        ! Triplet errors: systematic errors are different in each MQX magnet [-10, 10]
        ! + Random B2R 
        select, flag=error, clear;
        select, flag=error, pattern = "^MQX[AB]\..*";
        B2r = 4;
        ON_B2R = 1;
        ! to make all triplets have a different B2S component
        B2sX = 10-20*RANF();
        ON_B2S = 1;
        Rr = 0.050;

        ! macro to assign systematic errors 
        !(with = instead of := the assgned errors are same for all selected magnets in the class)
        SetEfcomp_QEL: macro = {
        Efcomp,  radius = Rr, order= 1,
                dknr:={0,
                1E-5*(B2sX*ON_B2S  + B2r*ON_B2R * TGAUSS(GCUTR))};
                }
        ! CHANGED 1E-4 is the usual, used 1E-5

        select, flag=error, clear;
        select, flag=error, pattern = "^MQX[AB]\..*";
        exec, SetEfcomp_QEL;

        ! Longitudinal misalignment of triplet quads (assumed to be 6mm)
        select, flag=error, clear;
        select, flag=error, pattern = "^MQX[AB]\..*";
        EALIGN, DS := 0.006*TGAUSS(3);

        ! save common triplet errors in a file, set in addition to individual errors
        select, flag=error, pattern = "^MQX[AB]\..*";
        etable, table="cetab"; ! Saving errors in table 

        !Assign average dipole errors (best knowldge model)
        readmytable, file = "/afs/cern.ch/eng/sl/lintrack/error_tables/Beam1/error_tables_6.5TeV/MBx-0001.errors", table=errtab;
        seterr, table=errtab;

        ! Save all assigned errors in one error table
        select, flag=error, clear;
        select, flag=error, pattern = "^MQ[^I^S^D].*";
        etable, table="etabb1"; ! Saving errors in table

        !exec, do_twiss_elements(LHCB1, "", 0.0);
        twiss, chrom, sequence=LHCB1, deltap=0.0, file="";
        '''% {"INDEX": str(index), "OPTICS": OPTICS, "SEED": seed})

    def match_tunes_b1(self):
        self.input('''
        print, text="Matching Tunes B1";
        exec, match_tunes(62.31, 60.32, 1);
        !exec, do_twiss_elements(LHCB1, "", 0.0);
        twiss, chrom, sequence=LHCB1, deltap=0.0, file="";

        !Maybe this second twiss is not needed

        ! Generate twiss with columns needed for training data
        ndx := table(twiss,dx)/sqrt(table(twiss,betx));
        select, flag=twiss, clear;
        select, flag=twiss, pattern="^BPM.*B1$", column=name, s, betx, bety, ndx,
                                                    mux, muy;
        twiss, chrom, sequence=LHCB1, deltap=0.0, file="";
        !./magnet_errors/b1_twiss_%(INDEX)s.tfs
        ! esave, file="./errors.tfs";
        ''')

    def job_magneterrors_b2(self, OPTICS, index, seed):
        self.input('''
        !@require lhc_runIII_2022.macros.madx        

        option, -echo;

        call, file = "/afs/cern.ch/eng/acc-models/lhc/2022/lhc.seq";

        exec, define_nominal_beams();
        !call, file="/afs/cern.ch/eng/acc-models/lhc/2022/operation/optics/R2023a_A30cmC30cmA10mL200cm.madx";
        call, file = "%(OPTICS)s";
        exec, cycle_sequences();

        use, period = LHCB2;

        option, echo;

        exec, match_tunes(62.31, 60.32, 2);

        ! generate individual errors for beam 2
        eoption, seed = %(SEED)s, add=true;
        ON_B2R = 1;
        GCUTR = 3; ! Cut for truncated gaussians (sigmas)

        !!!! Global errors !!!!
        select, flag=error, clear;
        select, flag=error, pattern = "^MQ\..*B2";
        Rr = 0.017;
        B2r = 19;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQM[LC]\..*B2";
        Rr = 0.017;
        B2r = 12;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQM\..*B2";
        Rr = 0.017;
        B2r = 12;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQY\..*B2";
        Rr = 0.017; // to be checked
        ! B2r = 8;
        B2r = 11;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQW[AB]\..*B2";
        Rr = 0.017;
        B2r = 15;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQW\..*B2";
        Rr = 0.017;
        B2r = 15;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQT\..*B2";
        Rr = 0.017;
        B2r = 15;
        exec, SetEfcomp_Q;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQTL[IH]\..*B2";
        Rr = 0.017;
        ! B2r = 15;
        B2r = 75;
        exec, SetEfcomp_Q;

        select, flag=error, clear;

        !Assigning the common error tab
        !READMYTABLE, file="./magnet_errors/common_errors_%(INDEX)s.tfs", table=errtab;
        SETERR, TABLE=cetab;

        !Assign average dipole errors (best knowldge model)
        readmytable, file = "/afs/cern.ch/eng/sl/lintrack/error_tables/Beam2/error_tables_6.5TeV/MBx-0001.errors", table=errtab;
        seterr, table=errtab;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQ[^B^I^S^D].*";
        etable, table="etabb2"; ! Saving errors in table 

        !exec, do_twiss_elements(LHCB2, "", 0.0);
        twiss, chrom, sequence=LHCB2, deltap=0.0, file="";

        '''% {"INDEX": str(index), "OPTICS": OPTICS, "SEED": seed})

    def match_tunes_b2(self):
        self.input('''
        print, text="Matching Tunes B2";
        exec, match_tunes(62.31, 60.32, 2);
        !exec, do_twiss_elements(LHCB2, "", 0.0);
        twiss, chrom, sequence=LHCB2, deltap=0.0, file="";

        !Maybe this second twiss is not needed

        ! Generate twiss with columns needed for training data
        ndx := table(twiss,dx)/sqrt(table(twiss,betx));
        select, flag=twiss, clear;
        select, flag=twiss, pattern="^BPM.*B2$", column=name, s, betx, bety, ndx,
                                                    mux, muy;
        twiss, chrom, sequence=LHCB2, deltap=0.0, file="";
        !esave, file="./errors.tfs";

        ''')

# %%
