#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
//#include <iostream>
//#include <lal/LALDatatypes.h>

#define ROTATEY(angle, vx, vy, vz)\
tmp1 = vx*cos(angle) + vz*sin(angle);\
tmp2 = - vx*sin(angle) + vz*cos(angle);\
vx = tmp1;\
vz = tmp2

#define ROTATEZ(angle, vx, vy, vz)\
tmp1 = vx*cos(angle) - vy*sin(angle);\
tmp2 = vx*sin(angle) + vy*cos(angle);\
vx = tmp1;\
vy = tmp2


#define REAL8 double
#define MAX_TOL_ATAN 1e-10
#define XLAL_EFAULT 0
#define XLAL_EDOM 0
#define LAL_MSUN_SI  1.988409902147041637325262574352366540e30
#define LAL_PI 3.14159265359
#define C 299792458.0
#define G 6.67430e-11
#define LAL_MTSUN_SI LAL_MSUN_SI*G/(C*C*C)
#define XLAL_EINVAL  0
#define XLAL_SUCCESS 0

double pow_2_of(double x){
  return x*x;
}

 static REAL8 atan2tol(REAL8 a, REAL8 b, REAL8 tol)
 {
   REAL8 c;
   if (fabs(a) < tol && fabs(b) < tol)
     c = 0.;
   else
     c = atan2(a, b);
   return c;
 }

 static REAL8 L2PNR_v1(
   const REAL8 v,   /**< Cubic root of (Pi * Frequency (geometric)) */
   const REAL8 eta) /**< Symmetric mass-ratio */
 {
   const REAL8 mu = eta; /* M=1 */
   const REAL8 v2 = v*v;
   const REAL8 v3 = v2*v;
   const REAL8 v4 = v3*v;
   const REAL8 eta2 = eta*eta;
   const REAL8 b = (4.75 + eta/9.)*eta*v4;
  
  
   return mu*sqrt((1 - ((3 - eta)*v2)/3. + b)/v2)*
     (1 + ((1 - 3*eta)*v2)/2. + (3*(1 - 7*eta + 13*eta2)*v4)/8. +
       ((14 - 41*eta + 4*eta2)*v4)/(4.*pow_2_of(1 - ((3 - eta)*v2)/3. + b)) +
       ((3 + eta)*v2)/(1 - ((3 - eta)*v2)/3. + b) +
       ((7 - 10*eta - 9*eta2)*v4)/(2.*(1 - ((3 - eta)*v2)/3. + b)));
 }

 /**
  * Simple 2PN version of the orbital angular momentum L,
  * without any spin terms expressed as a function of v.
  * For IMRPhenomP(v2).
  *
  *  Reference:
  *  - Boh&eacute; et al, 1212.5520v2 Eq 4.7 first line
  */
 static REAL8 L2PNR(
   const REAL8 v,   /**< Cubic root of (Pi * Frequency (geometric)) */
   const REAL8 eta) /**< Symmetric mass-ratio */
 {
   const REAL8 eta2 = eta*eta;
   const REAL8 x = v*v;
   const REAL8 x2 = x*x;
   return (eta*(1.0 + (1.5 + eta/6.0)*x + (3.375 - (19.0*eta)/8. - eta2/24.0)*x2)) / sqrt(x);
 }

void printstr(char *string){
  for (size_t i = 0; i < strlen(string); i++) {
    // Access each char in the string
    printf("%c",string[i]);
  }
}

void XLAL_ERROR(int EINVAL, char *message){
    printstr(message);
    printf("\n");
}

void XLAL_CHECK(int statement, int fault, char *message){
    if (statement =0){
      printf("error \n");
      printstr(message);
    }
}

typedef enum tagIMRPhenomP_version_type {
  IMRPhenomPv1_V, /**< version 1: based on IMRPhenomC */
  IMRPhenomPv2_V,  /**< version 2: based on IMRPhenomD */
  IMRPhenomPv2NRTidal_V, /**< version Pv2_NRTidal: based on IMRPhenomPv2; NRTides added before precession; can be used with both NRTidal versions defined below */
  IMRPhenomPv3_V  /**< version 3: based on IMRPhenomD and the precession angles from Katerina Chatziioannou PhysRevD.95.104004 (arxiv:1703.03967) */
 } IMRPhenomP_version_type;

/**
  * Function to map LAL parameters
  * (masses, 6 spin components, phiRef and inclination at f_ref)
  * (assumed to be in the source frame
  *  where LN points in the z direction
  *  i.e. lnhat = (0,0,1)
  *  and the separation vector n is in the x direction
  *  and the spherical angles of the line of sight N are (incl,Pi/2-phiRef))
  * into IMRPhenomP intrinsic parameters
  * (chi1_l, chi2_l, chip, thetaJN, alpha0 and phi_aligned).
  *
  * All input masses and frequencies should be in SI units.
  *
  * See Fig. 1. in arxiv:1408.1810 for a diagram of the angles.
  */
 int XLALSimIMRPhenomPCalculateModelParametersFromSourceFrame(
     REAL8 *chi1_l,                  /**< [out] Dimensionless aligned spin on companion 1 */
     REAL8 *chi2_l,                  /**< [out] Dimensionless aligned spin on companion 2 */
     REAL8 *chip,                    /**< [out] Effective spin in the orbital plane */
     REAL8 *thetaJN,                  /**< [out] Angle between J0 and line of sight (z-direction) */
     REAL8 *alpha0,                  /**< [out] Initial value of alpha angle (azimuthal precession angle) */
     REAL8 *phi_aligned,                  /**< [out] Initial phase to feed the underlying aligned-spin model */
     REAL8 *zeta_polariz,                  /**< [out] Angle to rotate the polarizations */
     const REAL8 m1_SI,              /**< Mass of companion 1 (kg) */
     const REAL8 m2_SI,              /**< Mass of companion 2 (kg) */
     const REAL8 f_ref,              /**< Reference GW frequency (Hz) */
     const REAL8 phiRef,              /**< Reference phase */
     const REAL8 incl,              /**< Inclination : angle between LN and the line of sight */
     const REAL8 s1x,                /**< Initial value of s1x: dimensionless spin of BH 1 */
     const REAL8 s1y,                /**< Initial value of s1y: dimensionless spin of BH 1 */
     const REAL8 s1z,                /**< Initial value of s1z: dimensionless spin of BH 1 */
     const REAL8 s2x,                /**< Initial value of s2x: dimensionless spin of BH 2 */
     const REAL8 s2y,                /**< Initial value of s2y: dimensionless spin of BH 2 */
     const REAL8 s2z,                /**< Initial value of s2z: dimensionless spin of  BH 2 */
     IMRPhenomP_version_type IMRPhenomP_version /**< IMRPhenomP(v1) uses IMRPhenomC, IMRPhenomPv2 uses IMRPhenomD, IMRPhenomPv2_NRTidal uses NRTidal framework with IMRPhenomPv2 */
 )
 {
   // Note that the angle phiJ defined below and alpha0 are degenerate. Therefore we do not output phiJ.
  
   /* Check arguments for sanity */
   XLAL_CHECK(chi1_l != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(chi2_l != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(chip != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(thetaJN != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(alpha0 != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(phi_aligned != NULL, XLAL_EFAULT,"");
  
   XLAL_CHECK(f_ref > 0, XLAL_EDOM, "Reference frequency must be positive.\n");
   XLAL_CHECK(m1_SI > 0, XLAL_EDOM, "m1 must be positive.\n");
   XLAL_CHECK(m2_SI > 0, XLAL_EDOM, "m2 must be positive.\n");
   XLAL_CHECK(fabs(s1x*s1x + s1y*s1y + s1z*s1z) <= 1.0, XLAL_EDOM, "|S1/m1^2| must be <= 1.\n");
   XLAL_CHECK(fabs(s2x*s2x + s2y*s2y + s2z*s2z) <= 1.0, XLAL_EDOM, "|S2/m2^2| must be <= 1.\n");
  
   const REAL8 m1 = m1_SI / LAL_MSUN_SI;   /* Masses in solar masses */
   const REAL8 m2 = m2_SI / LAL_MSUN_SI;
   const REAL8 M = m1+m2;
   const REAL8 m1_2 = m1*m1;
   const REAL8 m2_2 = m2*m2;
   const REAL8 eta = m1 * m2 / (M*M);    /* Symmetric mass-ratio */
  
   /* From the components in the source frame, we can easily determine
    chi1_l, chi2_l, chip and phi_aligned, which we need to return.
    We also compute the spherical angles of J,
    which we need to transform to the J frame*/
  
   /* Aligned spins */
   *chi1_l = s1z; /* Dimensionless aligned spin on BH 1 */
   *chi2_l = s2z; /* Dimensionless aligned spin on BH 2 */
  
   /* Magnitude of the spin projections in the orbital plane */
   const REAL8 S1_perp = m1_2*sqrt(s1x*s1x + s1y*s1y);
   const REAL8 S2_perp = m2_2*sqrt(s2x*s2x + s2y*s2y);

   //printf("perps: %lf %lf \n", S1_perp, S2_perp);

   /* From this we can compute chip*/
   const REAL8 A1 = 2 + (3*m2) / (2*m1);
   const REAL8 A2 = 2 + (3*m1) / (2*m2);
   const REAL8 ASp1 = A1*S1_perp;
   const REAL8 ASp2 = A2*S2_perp;
   const REAL8 num = (ASp2 > ASp1) ? ASp2 : ASp1;
   const REAL8 den = (m2 > m1) ? A2*m2_2 : A1*m1_2;
   *chip = num / den; /*  chip = max(A1 Sp1, A2 Sp2) / (A_i m_i^2) for i index of larger BH (See Eqn. 32 in technical document) */
  
   /* Compute L, J0 and orientation angles */
   const REAL8 m_sec = M * LAL_MTSUN_SI;   /* Total mass in seconds */
   const REAL8 piM = LAL_PI * m_sec;
   //printf("piM: %lf \n", piM );
   const REAL8 v_ref = cbrt(piM * f_ref);
  
   const int ExpansionOrder = 5; // Used in PhenomPv3 only
  
   REAL8 L0 = 0.0;
   switch (IMRPhenomP_version) {
     case IMRPhenomPv1_V:
       printf("I am here v1\n");

       L0 = M*M * L2PNR_v1(v_ref, eta); /* Use 2PN approximation for L. */
       break;
     case IMRPhenomPv2_V:
     case IMRPhenomPv2NRTidal_V:
       printf("I am here v2\n");
       L0 = M*M * L2PNR(v_ref, eta);   /* Use 2PN approximation for L. */
       break;
     case IMRPhenomPv3_V: /*Pv3 uses 3PN spinning for L but in non-precessing limit uses the simpler L2PNR function */
       printf("I am here v3\n");
       if ((s1x == 0. && s1y == 0. && s2x == 0. && s2y == 0.))
       { // non-precessing case
         L0 = M * M * L2PNR(v_ref, eta); /* Use 2PN approximation for L. */
       } else { // precessing case
         //L0 = M * M * PhenomInternal_OrbAngMom3PN(f_ref / 2., m1_SI, m2_SI, s1x, s1y, s1z, s2x, s2y, s2z, f_ref, ExpansionOrder); /* Use 3PN spinning approximation for L. */
          L0=0;
       }
       break;
     default:
       XLAL_ERROR( XLAL_EINVAL, "Unknown IMRPhenomP version!\nAt present only v1 and v2 are available." );
       break;
     }
   //printf("L0 input: %.10f %.10f %.10f \n", v_ref, eta, M);
   //printf("L0: %.10f \n", L0);
   // Below, _sf indicates source frame components. We will also use _Jf for J frame components
   const REAL8 J0x_sf = m1_2*s1x + m2_2*s2x;
   const REAL8 J0y_sf = m1_2*s1y + m2_2*s2y;
   const REAL8 J0z_sf = L0 + m1_2*s1z + m2_2*s2z;
   const REAL8 J0 = sqrt(J0x_sf*J0x_sf + J0y_sf*J0y_sf + J0z_sf*J0z_sf);
  
   /* Compute thetaJ, the angle between J0 and LN (z-direction) */
   REAL8 thetaJ_sf;
   if (J0 < 1e-10) {
     //XLAL_PRINT_WARNING("Warning: |J0| < 1e-10. Setting thetaJ = 0.\n");
     //printf("Warning: |J0| < 1e-10. Setting thetaJ = 0.\n");
     thetaJ_sf = 0;
   } else {
     thetaJ_sf = acos(J0z_sf / J0);
   }
  
   REAL8 phiJ_sf;
   if (fabs(J0x_sf) < MAX_TOL_ATAN && fabs(J0y_sf) < MAX_TOL_ATAN)
     phiJ_sf = LAL_PI/2. - phiRef; // aligned spin limit
   else
     phiJ_sf = atan2(J0y_sf, J0x_sf); /* azimuthal angle of J0 in the source frame */
  
   *phi_aligned = - phiJ_sf;
  
   /* We now have to rotate to the "J frame" where we can easily
    compute alpha0, the azimuthal angle of LN,
    as well as thetaJ, the angle between J and N.
    The J frame is defined imposing that J points in the z direction
    and the line of sight N is in the xz plane (with positive projection along x).
    The components of any vector in the (new) J frame are obtained from those
    in the (old) source frame by multiplying by RZ[kappa].RY[-thetaJ].RZ[-phiJ]
    where kappa will be determined by rotating N with RY[-thetaJ].RZ[-phiJ]
    (which brings J to the z axis) and taking the opposite of azimuthal angle of the rotated N.
    */
   REAL8 tmp1,tmp2;
   // First we determine kappa
   // in the source frame, the components of N are given in Eq (35c) of T1500606-v6
   REAL8 Nx_sf = sin(incl)*cos(LAL_PI/2. - phiRef);
   REAL8 Ny_sf = sin(incl)*sin(LAL_PI/2. - phiRef);
   REAL8 Nz_sf = cos(incl);
   REAL8 tmp_x = Nx_sf;
   REAL8 tmp_y = Ny_sf;
   REAL8 tmp_z = Nz_sf;
   ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z);
   REAL8 kappa;
   kappa = - atan2tol(tmp_y,tmp_x, MAX_TOL_ATAN);
   //printf("%lf,\n",kappa);
  
   // Then we determine alpha0, by rotating LN
   tmp_x = 0.;
   tmp_y = 0.;
   tmp_z = 1.; // in the source frame, LN=(0,0,1)
   ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEZ(kappa, tmp_x, tmp_y, tmp_z);
   //printf("%lf, %lf \n",tmp_x,tmp_y);
   if (fabs(tmp_x) < MAX_TOL_ATAN && fabs(tmp_y) < MAX_TOL_ATAN)
     *alpha0 = LAL_PI; //this is the aligned spin case
   else
     *alpha0 = atan2(tmp_y,tmp_x);
  
   // Finally we determine thetaJ, by rotating N
   tmp_x = Nx_sf;
   tmp_y = Ny_sf;
   tmp_z = Nz_sf;
   ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEZ(kappa, tmp_x, tmp_y, tmp_z);
   REAL8 Nx_Jf = tmp_x; // let's store those two since we will reuse them later (we don't need the y component)
   REAL8 Nz_Jf = tmp_z;
   *thetaJN = acos(Nz_Jf); // No normalization needed, we are dealing with a unit vector
  
   /* Finally, we need to redefine the polarizations :
    PhenomP's polarizations are defined following Arun et al (arXiv:0810.5336)
    i.e. projecting the metric onto the P,Q,N triad defined with P=NxJ/|NxJ| (see (2.6) in there).
    By contrast, the triad X,Y,N used in LAL
    ("waveframe" in the nomenclature of T1500606-v6)
    is defined in e.g. eq (35) of this document
    (via its components in the source frame; note we use the defautl Omega=Pi/2).
    Both triads differ from each other by a rotation around N by an angle \zeta
    and we need to rotate the polarizations accordingly by 2\zeta
   */
   REAL8 Xx_sf = -cos(incl)*sin(phiRef);
   REAL8 Xy_sf = -cos(incl)*cos(phiRef);
   REAL8 Xz_sf = sin(incl);
   tmp_x = Xx_sf;
   tmp_y = Xy_sf;
   tmp_z = Xz_sf;
   ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEZ(kappa, tmp_x, tmp_y, tmp_z);
   //now the tmp_a are the components of X in the J frame
   //we need the polar angle of that vector in the P,Q basis of Arun et al
   // P=NxJ/|NxJ| and since we put N in the (pos x)z half plane of the J frame
   REAL8 PArunx_Jf = 0.;
   REAL8 PAruny_Jf = -1.;
   REAL8 PArunz_Jf = 0.;
   // Q=NxP
   REAL8 QArunx_Jf = Nz_Jf;
   REAL8 QAruny_Jf = 0.;
   REAL8 QArunz_Jf = -Nx_Jf;
   REAL8 XdotPArun = tmp_x*PArunx_Jf+tmp_y*PAruny_Jf+tmp_z*PArunz_Jf;
   REAL8 XdotQArun = tmp_x*QArunx_Jf+tmp_y*QAruny_Jf+tmp_z*QArunz_Jf;
   *zeta_polariz = atan2(XdotQArun , XdotPArun);
  
   return XLAL_SUCCESS;
 }


 typedef struct tagNNLOanglecoeffs {
     REAL8 alphacoeff1; /* Coefficient of omega^(-1)   in alphaNNLO */
     REAL8 alphacoeff2; /* Coefficient of omega^(-2/3) in alphaNNLO */
     REAL8 alphacoeff3; /* Coefficient of omega^(-1/3) in alphaNNLO */
     REAL8 alphacoeff4; /* Coefficient of log(omega)   in alphaNNLO */
     REAL8 alphacoeff5; /* Coefficient of omega^(1/3)  in alphaNNLO */
  
     REAL8 epsiloncoeff1; /* Coefficient of omega^(-1)   in epsilonNNLO */
     REAL8 epsiloncoeff2; /* Coefficient of omega^(-2/3) in epsilonNNLO */
     REAL8 epsiloncoeff3; /* Coefficient of omega^(-1/3) in epsilonNNLO */
     REAL8 epsiloncoeff4; /* Coefficient of log(omega)   in epsilonNNLO */
     REAL8 epsiloncoeff5; /* Coefficient of omega^(1/3)  in epsilonNNLO */
 } NNLOanglecoeffs;
 /**
  * Next-to-next-to-leading order PN coefficients
  * for Euler angles \f$\alpha\f$ and \f$\epsilon\f$.
  */

 static void ComputeNNLOanglecoeffs(
   NNLOanglecoeffs *angcoeffs, /**< [out] Structure to store results */
   const REAL8 q,              /**< Mass-ratio (convention q>1) */
   const REAL8 chil,           /**< Dimensionless aligned spin of the largest BH */
   const REAL8 chip)           /**< Dimensionless spin component in the orbital plane */
 {
   const REAL8 m2 = q/(1. + q);
   const REAL8 m1 = 1./(1. + q);
   const REAL8 dm = m1 - m2;
   const REAL8 mtot = 1.;
   const REAL8 eta = m1*m2; /* mtot = 1 */
   const REAL8 eta2 = eta*eta;
   const REAL8 eta3 = eta2*eta;
   const REAL8 eta4 = eta3*eta;
   const REAL8 mtot2 = mtot*mtot;
   const REAL8 mtot4 = mtot2*mtot2;
   const REAL8 mtot6 = mtot4*mtot2;
   const REAL8 mtot8 = mtot6*mtot2;
   const REAL8 chil2 = chil*chil;
   const REAL8 chip2 = chip*chip;
   const REAL8 chip4 = chip2*chip2;
   const REAL8 dm2 = dm*dm;
   const REAL8 dm3 = dm2*dm;
   const REAL8 m2_2 = m2*m2;
   const REAL8 m2_3 = m2_2*m2;
   const REAL8 m2_4 = m2_3*m2;
   const REAL8 m2_5 = m2_4*m2;
   const REAL8 m2_6 = m2_5*m2;
   const REAL8 m2_7 = m2_6*m2;
   const REAL8 m2_8 = m2_7*m2;
  
  
   angcoeffs->alphacoeff1 = (-0.18229166666666666 - (5*dm)/(64.*m2));
  
   angcoeffs->alphacoeff2 = ((-15*dm*m2*chil)/(128.*mtot2*eta) - (35*m2_2*chil)/(128.*mtot2*eta));
  
   angcoeffs->alphacoeff3 = (-1.7952473958333333 - (4555*dm)/(7168.*m2) -
         (15*chip2*dm*m2_3)/(128.*mtot4*eta2) -
         (35*chip2*m2_4)/(128.*mtot4*eta2) - (515*eta)/384. - (15*dm2*eta)/(256.*m2_2) -
         (175*dm*eta)/(256.*m2));
  
   angcoeffs->alphacoeff4 = - (35*LAL_PI)/48. - (5*dm*LAL_PI)/(16.*m2) +
      (5*dm2*chil)/(16.*mtot2) + (5*dm*m2*chil)/(3.*mtot2) +
      (2545*m2_2*chil)/(1152.*mtot2) -
      (5*chip2*dm*m2_5*chil)/(128.*mtot6*eta3) -
      (35*chip2*m2_6*chil)/(384.*mtot6*eta3) + (2035*dm*m2*chil)/(21504.*mtot2*eta) +
      (2995*m2_2*chil)/(9216.*mtot2*eta);
  
   angcoeffs->alphacoeff5 = (4.318908476114694 + (27895885*dm)/(2.1676032e7*m2) -
         (15*chip4*dm*m2_7)/(512.*mtot8*eta4) -
         (35*chip4*m2_8)/(512.*mtot8*eta4) -
         (485*chip2*dm*m2_3)/(14336.*mtot4*eta2) +
         (475*chip2*m2_4)/(6144.*mtot4*eta2) +
         (15*chip2*dm2*m2_2)/(256.*mtot4*eta) + (145*chip2*dm*m2_3)/(512.*mtot4*eta) +
         (575*chip2*m2_4)/(1536.*mtot4*eta) + (39695*eta)/86016. + (1615*dm2*eta)/(28672.*m2_2) -
         (265*dm*eta)/(14336.*m2) + (955*eta2)/576. + (15*dm3*eta2)/(1024.*m2_3) +
         (35*dm2*eta2)/(256.*m2_2) + (2725*dm*eta2)/(3072.*m2) - (15*dm*m2*LAL_PI*chil)/(16.*mtot2*eta) -
         (35*m2_2*LAL_PI*chil)/(16.*mtot2*eta) + (15*chip2*dm*m2_7*chil2)/(128.*mtot8*eta4) +
         (35*chip2*m2_8*chil2)/(128.*mtot8*eta4) +
         (375*dm2*m2_2*chil2)/(256.*mtot4*eta) + (1815*dm*m2_3*chil2)/(256.*mtot4*eta) +
         (1645*m2_4*chil2)/(192.*mtot4*eta));
  
   angcoeffs->epsiloncoeff1 = (-0.18229166666666666 - (5*dm)/(64.*m2));
  
   angcoeffs->epsiloncoeff2 = ((-15*dm*m2*chil)/(128.*mtot2*eta) - (35*m2_2*chil)/(128.*mtot2*eta));
  
   angcoeffs->epsiloncoeff3 = (-1.7952473958333333 - (4555*dm)/(7168.*m2) - (515*eta)/384. -
         (15*dm2*eta)/(256.*m2_2) - (175*dm*eta)/(256.*m2));
  
   angcoeffs->epsiloncoeff4 = - (35*LAL_PI)/48. - (5*dm*LAL_PI)/(16.*m2) +
      (5*dm2*chil)/(16.*mtot2) + (5*dm*m2*chil)/(3.*mtot2) +
      (2545*m2_2*chil)/(1152.*mtot2) + (2035*dm*m2*chil)/(21504.*mtot2*eta) +
      (2995*m2_2*chil)/(9216.*mtot2*eta);
  
   angcoeffs->epsiloncoeff5 = (4.318908476114694 + (27895885*dm)/(2.1676032e7*m2) + (39695*eta)/86016. +
         (1615*dm2*eta)/(28672.*m2_2) - (265*dm*eta)/(14336.*m2) + (955*eta2)/576. +
         (15*dm3*eta2)/(1024.*m2_3) + (35*dm2*eta2)/(256.*m2_2) +
         (2725*dm*eta2)/(3072.*m2) - (15*dm*m2*LAL_PI*chil)/(16.*mtot2*eta) - (35*m2_2*LAL_PI*chil)/(16.*mtot2*eta) +
         (375*dm2*m2_2*chil2)/(256.*mtot4*eta) + (1815*dm*m2_3*chil2)/(256.*mtot4*eta) +
         (1645*m2_4*chil2)/(192.*mtot4*eta));
 }

int main(){
    REAL8 tmp1, tmp2;
    REAL8 chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz;
    REAL8 m1_SI, m2_SI, f_ref, phiRef, incl, s1x, s1y, s1z, s2x, s2y, s2z;
    m1_SI = 3e30;
    m2_SI = 4e30;
    f_ref = 30;
    phiRef = 0.3;
    incl = 0.1;
    s1x = 0.3;
    s1y = 0;
    s1z = 0.34;
    s2x = 0.1;
    s2y = 0.1;
    s2z = 0.5;

    //IMRPhenomP_version_type IMRPhenomPv2_V;

    XLALSimIMRPhenomPCalculateModelParametersFromSourceFrame(&chi1_l,&chi2_l, &chip, &thetaJN, &alpha0, &phi_aligned, &zeta_polariz, m1_SI, m2_SI, f_ref, phiRef, incl, s1x, s1y, s1z, s2x, s2y, s2z, IMRPhenomPv2_V);
    

    NNLOanglecoeffs angcoeffs; /* Next-to-next-to leading order PN coefficients for Euler angles alpha and epsilon */
    const REAL8 m1 = m1_SI / LAL_MSUN_SI;
    const REAL8 m2 = m2_SI / LAL_MSUN_SI;
    const REAL8 M = m1 + m2;
    const REAL8 m_sec = M * LAL_MTSUN_SI;   /* Total mass in seconds */
    REAL8 q = m2 / m1; /* q >= 1 */
    const REAL8 chi_eff = (m1*chi1_l + m2*chi2_l) / M; /* Effective aligned spin */
    const REAL8 chil = (1.0+q)/q * chi_eff; /* dimensionless aligned spin of the largest BH */
    ComputeNNLOanglecoeffs(&angcoeffs,q,chil,chip);
    printf("%.10f, %.10f, %.10f, %.10f, %.10f, %.10f %.10f", chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz);
    printf("\n");  

    printf("%.10f, %.10f, %.10f, %.10f, %.10f \n", 
    angcoeffs.alphacoeff1, angcoeffs.alphacoeff2, angcoeffs.alphacoeff3, 
    angcoeffs.alphacoeff4, angcoeffs.alphacoeff5);
    
    printf("%.10f, %.10f, %.10f, %.10f, %.10f \n", 
    angcoeffs.epsiloncoeff1, angcoeffs.epsiloncoeff2, angcoeffs.epsiloncoeff3, 
    angcoeffs.epsiloncoeff4, angcoeffs.epsiloncoeff5);

    return 0;
}