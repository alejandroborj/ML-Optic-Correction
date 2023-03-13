#%%
import cpymad.madx


class madx_ml_op(cpymad.madx.Madx):
    '''Normal cpymad wrapper with extra methods for this exact project'''

    def job_magneterrors_b1(self, OPTICS, index, seed):
        self.input('''
        option, -echo;
        call, file = "./afs/beta_beat.macros.madx";
        !"/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx";
        call, file = "./afs/lhc.macros.madx";
        !"/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx";
        option, echo;
        
        ! in the litrack macros kqts are defined wrong for 2016 optics
        match_tunes_kqt(nqx, nqy, beam_number): macro = {
            match;
            vary, name=KQTD.Bbeam_number;
            vary, name=KQTF.Bbeam_number;
            GLOBAL, Q1= nqx, Q2=nqy;
            lmdif, calls=2000, tolerance=1E-23;
            endmatch;
        };

        call, file = "./afs/Esubroutines.madx";
        !"/afs/cern.ch/eng/lhc/optics/V6.5/errors/Esubroutines.madx";

        ! load main sequence
        option, -echo;
        call, file = "./afs/main.seq";
        !"/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/model/accelerators/lhc/2016/main.seq";

        beam, sequence=LHCB1, particle=proton, energy=6500, kbunch=1, npart=1.15E11, bv=1;
        call, file = "%(OPTICS)s";
        exec, cycle_sequences();
        use, period = LHCB1;

        option, echo;
        exec, match_tunes_kqt(64.28, 59.31, 1);

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
                1E-4*(B2sX*ON_B2S  + B2r*ON_B2R * TGAUSS(GCUTR))};
                }
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
        esave, file="./magnet_errors/common_errors_%(INDEX)s.tfs";

        ! Add sextupole misalignments: 
        ! --> not needed anymore, because MQ arcs B2R is increased by 1 unit.
        ! select, flag=error, clear;
        ! SELECT, FLAG = ERROR, PATTERN = "^MS\..*B1$";
        ! EALIGN, DX := 0.0003*TGAUSS(3);

        ! ! Add logitudinal misalignments to arc quads, 
        ! TODO for later if MQX misalignment can be predicted well
        ! select, flag=error, clear;
        ! select, flag=error, pattern = "^MQ[^I^S^D].*B1$";
        ! EALIGN, DS := 0.006*TGAUSS(3);

        !Assign average dipole errors (best knowldge model)
        readmytable, file = "./afs/MBx-0001.errors", table=errtab;
        seterr, table=errtab;
        !readmytable, file = "/afs/cern.ch/eng/sl/lintrack/error_tables/Beam1/error_tables_6.5TeV/MBx-0001.errors", table=errtab;

        ! Save all assigned errors in one error table
        select, flag=error, clear;
        select, flag=error, pattern = "^MQ[^I^S^D].*";
        etable, table="etabb1"; ! Saving errors in table 
        !esave, file="./magnet_errors/b1_errors_%(INDEX)s.tfs";

        exec, do_twiss_elements(LHCB1, "./magnet_errors/b1_twiss_before_match_%(INDEX)s.tfs", 0.0);
        exec, match_tunes_kqt(64.28, 59.31, 1);
        exec, do_twiss_elements(LHCB1, "./magnet_errors/b1_twiss_after_match_%(INDEX)s.tfs", 0.0);

        ! Generate twiss with columns needed for training data
        ndx := table(twiss,dx)/sqrt(table(twiss,betx));
        select, flag=twiss, clear;
        select, flag=twiss, pattern="^BPM.*B1$", column=name, s, betx, bety, ndx,
                                                    mux, muy;
        twiss, chrom, sequence=LHCB1, deltap=0.0, file="./magnet_errors/b1_twiss_%(INDEX)s.tfs";
        ''' % {"INDEX": str(index), "OPTICS": OPTICS, "SEED": seed})


    def job_magneterrors_b2(self, OPTICS, index, seed):
        self.input('''
        option, -echo;
        call, file = "./afs/beta_beat.macros.madx";
        !call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx";
        call, file = "./afs/lhc.macros.madx";
        !call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx";
        option, echo;

        match_tunes_kqt(nqx, nqy, beam_number): macro = {
            match;
            vary, name=KQTD.Bbeam_number;
            vary, name=KQTF.Bbeam_number;
            GLOBAL, Q1= nqx, Q2=nqy;
            lmdif, calls=2000, tolerance=1E-23;
            endmatch;
        };
        call, file = "./afs/Esubroutines.madx";
        !call, file="/afs/cern.ch/eng/lhc/optics/V6.5/errors/Esubroutines.madx";
        call, file = "./afs/main.seq";
        !call, file = "/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/model/accelerators/lhc/2016/main.seq";
        beam, sequence=LHCB2, particle=proton, energy=6500, kbunch=1, npart=1.15E11, bv=-1;
        call, file = "%(OPTICS)s";
        exec, cycle_sequences();
        use, period = LHCB2;

        option, echo;
        exec, match_tunes_kqt(64.28, 59.31, 2);

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
        READMYTABLE, file="./magnet_errors/common_errors_%(INDEX)s.tfs", table=errtab;
        SETERR, TABLE=errtab;

        ! Add sextupole misalignments:
        ! select, flag=error, clear;
        ! SELECT, FLAG = ERROR, PATTERN = "^MS\..*B2$";
        ! EALIGN, DX := 0.0003*TGAUSS(3);

        ! Add quads longitudinal misalignments:
        ! select, flag=error, clear;
        ! select, flag=error, pattern = "^MQ[^I^S^D].*B2$";
        ! EALIGN, DS := 0.006*TGAUSS(3);

        !Assign average dipole errors (best knowldge model)
        !readmytable, file = "/afs/cern.ch/eng/sl/lintrack/error_tables/Beam2/error_tables_6.5TeV/MBx-0001.errors", table=errtab;
        readmytable, file = "./afs/MBx-0001.errors", table=errtab;
        seterr, table=errtab;

        select, flag=error, clear;
        select, flag=error, pattern = "^MQ[^B^I^S^D].*";
        etable, table="etabb2"; ! Saving errors in table 
        !esave, file="./magnet_errors/b2_errors_%(INDEX)s.tfs";

        exec, do_twiss_elements(LHCB2, "./magnet_errors/b2_twiss_before_match_%(INDEX)s.tfs", 0.0);
        exec, match_tunes_kqt(64.28, 59.31, 2);
        exec, do_twiss_elements(LHCB2, "./magnet_errors/b2_twiss_after_match_%(INDEX)s.tfs", 0.0);

        ndx := table(twiss,dx)/sqrt(table(twiss,betx));
        select, flag=twiss, clear;
        select, flag=twiss, pattern="^BPM.*B2$", column=name, s, betx, bety, ndx,
                                                    mux, muy;
        twiss, chrom, sequence=LHCB2, deltap=0.0, file="./magnet_errors/b2_twiss_%(INDEX)s.tfs";
        ''' % {"INDEX": str(index), "OPTICS": OPTICS, "SEED": seed})
# %%
