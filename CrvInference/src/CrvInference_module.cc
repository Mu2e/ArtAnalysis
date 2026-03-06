//
// Run XGBoost-based inference on CRV coincidence clusters
// paired with reconstructed tracks, producing an MVAResult
// score for each (KalSeed, CrvCoincidenceCluster) pair.
//
// Based on Andy Edmonds' TrackQuality_module.cc 
// Sam Grant 2026

// art
#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "fhiclcpp/types/Atom.h"
#include "canvas/Persistency/Common/Assns.h"

// Offline
#include "Offline/RecoDataProducts/inc/KalSeed.hh"
#include "Offline/RecoDataProducts/inc/CrvCoincidenceCluster.hh"
#include "Offline/RecoDataProducts/inc/MVAResult.hh"
#include "Offline/DataProducts/inc/SurfaceId.hh"
#include "Offline/ConfigTools/inc/ConfigFileLookupPolicy.hh"

// TMVA
#include "TMVA/RBDT.hxx"

// C++
#include <iostream>
#include <string>
#include <vector>

namespace mu2e {

  // assns to link objects within art::Event
  using CrvInferenceAssns = art::Assns<KalSeed, CrvCoincidenceCluster, MVAResult>;

  class CrvInference : public art::EDProducer {

  public:
    struct Config {
      using Name = fhicl::Name;
      using Comment = fhicl::Comment;

      fhicl::Atom<art::InputTag> kalSeedPtrTag{Name("KalSeedPtrCollection"), Comment("Input tag for KalSeedPtrCollection")};
      fhicl::Atom<art::InputTag> crvCoincidenceTag{Name("CrvCoincidenceClusterCollection"), Comment("Input tag for CrvCoincidenceClusterCollection")};
      fhicl::Atom<std::string> modelFileName{Name("modelFileName"), Comment("Path to RBDT .root model file")}; // 
      fhicl::Atom<std::string> modelKey{Name("modelKey"), Comment("Key name within the ROOT file")};
      fhicl::Atom<int> debug{Name("debugLevel"), Comment("Debug printout level"), 0};
    };

    using Parameters = art::EDProducer::Table<Config>;
    explicit CrvInference(const Parameters& conf);

  private:
    void produce(art::Event& event) override;

    art::InputTag _kalSeedPtrTag;
    art::InputTag _crvCoincidenceTag;
    int _debug;

    TMVA::Experimental::RBDT _bdt;

    static constexpr size_t _nFeatures = 9;
  };

  CrvInference::CrvInference(const Parameters& conf) :
    art::EDProducer{conf},
    _kalSeedPtrTag(conf().kalSeedPtrTag()),
    _crvCoincidenceTag(conf().crvCoincidenceTag()),
    _debug(conf().debug()),
    _bdt(conf().modelKey(), ConfigFileLookupPolicy()(conf().modelFileName()))
  {
    produces<CrvInferenceAssns>();
  }

  void CrvInference::produce(art::Event& event) {

    auto assns = std::make_unique<CrvInferenceAssns>();

    // get inputs
    const auto& kalSeedPtrHandle = event.getValidHandle<KalSeedPtrCollection>(_kalSeedPtrTag);
    const auto& crvCoincHandle = event.getValidHandle<CrvCoincidenceClusterCollection>(_crvCoincidenceTag);

    for (size_t i_ks = 0; i_ks < kalSeedPtrHandle->size(); ++i_ks) {
      const auto& kalSeedPtr = kalSeedPtrHandle->at(i_ks);
      const auto& kalSeed = *kalSeedPtr;

      // find tracker mid and 
      // extract track time
      double trkTime = -9999;
      bool ttMidFound = false;
      for (const auto& kinter : kalSeed.intersections()) {
        if (kinter.surfaceId() == SurfaceIdDetail::TT_Mid) {
          trkTime = kinter.time();
          ttMidFound = true;
          break;
        }
      }

      for (size_t i_crv = 0; i_crv < crvCoincHandle->size(); ++i_crv) {
        const auto& cluster = crvCoincHandle->at(i_crv);

        // build feature vector
        // hardcoded order - must match training!
        std::vector<float> features(_nFeatures);
        features[0] = cluster.GetAvgHitPos().x(); // CRV x-position
        features[1] = cluster.GetAvgHitPos().y(); // CRV y-position
        features[2] = cluster.GetAvgHitPos().z(); // CRV z-position
        features[3] = cluster.GetPEs(); // CRV total PEs
        features[4] = trkTime - cluster.GetAvgHitTime(); // Track-CRV time difference
        features[5] = cluster.GetCrvRecoPulses().size(); // Number of hits (reco pulses) in the cluster
        features[6] = cluster.GetLayers().size(); // Number of layers hit
        features[7] = cluster.GetSlope(); // Angle 
        features[8] = cluster.GetCrvSectorType(); // Sector

        // run the inference
        auto output = _bdt.Compute(features);
        float score = output[0];

        if (!ttMidFound) {
          score = 0;
        }

        if (_debug > 0) {
          std::cout << std::string(50, '=') << std::endl
                    << "[CrvInference] Run " << event.id().run()
                    << ", Subrun " << event.id().subRun()
                    << ", Event " << event.id().event() << std::endl
                    << "  nKalSeeds = " << kalSeedPtrHandle->size()
                    << ", nCrvCoincs = " << crvCoincHandle->size() << std::endl
                    << "  KalSeed " << i_ks << ", CrvCoinc " << i_crv << std::endl
                    << "  crv_x = " << features[0]
                    << ", crv_y = " << features[1]
                    << ", crv_z = " << features[2] << std::endl
                    << "  PEs = " << features[3]
                    << ", dt = " << features[4] << std::endl
                    << "  nHits = " << features[5]
                    << ", nLayers = " << features[6] << std::endl
                    << "  slope = " << features[7]
                    << ", sector = " << (int)features[8] << std::endl
                    << "  score = " << score << std::endl;
        }

        auto crvCoincPtr = art::Ptr<CrvCoincidenceCluster>(crvCoincHandle, i_crv);
        assns->addSingle(kalSeedPtr, crvCoincPtr, MVAResult(score));
      }
    }

    event.put(std::move(assns));
  }

} // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvInference)
