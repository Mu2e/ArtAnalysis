// Classify clusters into background and signal-like based
// on the process code of most of the particles in the cluster

#ifndef TrkDiag_BkgMCMatch_HH
#define TrkDiag_BkgMCMatch_HH

#include "Offline/MCDataProducts/inc/ProcessCode.hh"
#include <set>

namespace mu2e {

  namespace BkgMCMatch {
   
    inline bool isBackground(ProcessCode code) {
      static const std::set<ProcessCode> bkgCodes = {
        ProcessCode::annihil,                 // 2
        ProcessCode::compt,                   // 12
        ProcessCode::conv,                    // 13
        ProcessCode::eBrem,                   // 16
        ProcessCode::eIoni,                   // 17
        ProcessCode::hIoni,                   // 21
        ProcessCode::ionIoni,                 // 23
        ProcessCode::muIoni,                  // 31
        ProcessCode::nCapture,                // 35
        ProcessCode::phot,                    // 40
        ProcessCode::mu2ePrimary,             // 56
        ProcessCode::hadElastic,              // 58
        ProcessCode::CoulombScat,             // 59
        ProcessCode::dInelastic,              // 72 
        ProcessCode::RadioactiveDecay,        // 74
        ProcessCode::hBertiniCaptureAtRest,   // 78
        ProcessCode::neutronInelastic,        // 97
        ProcessCode::protonInelastic,         // 101
        ProcessCode::photonNuclear,           // 109
        ProcessCode::EMCascade,               // 113
        ProcessCode::DIO,                     // 114
        ProcessCode::NuclearCapture,          // 115
        ProcessCode::mu2eMuonCaptureAtRest,   // 165
        ProcessCode::mu2eMuonDecayAtRest      // 166
      };
      return bkgCodes.find(code) != bkgCodes.end();
    }

    inline bool isCE(ProcessCode code) {
      return code == ProcessCode::mu2eCeMinusEndpoint; // 167
    }

    inline int classify(ProcessCode code) {
      if (isCE(code)) return 1;
      if (isBackground(code)) return -2;
      return 0;
    }
  }

}

#endif
