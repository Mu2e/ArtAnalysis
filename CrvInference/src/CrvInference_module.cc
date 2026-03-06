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

  using CrvInferenceAssns = art::Assns<KalSeed, CrvCoincidenceCluster, MVAResult>;

  class CrvInference : public art::EDProducer {

  public:
    struct Config {
      using Name = fhicl::Name;
      using Comment = fhicl::Comment;

      fhicl::Atom<art::InputTag> kalSeedPtrTag{Name("KalSeedPtrCollection"), Comment("Input tag for KalSeedPtrCollection")};
      fhicl::Atom<art::InputTag> crvCoincidenceTag{Name("CrvCoincidenceClusterCollection"), Comment("Input tag for CrvCoincidenceClusterCollection")};
      fhicl::Atom<std::string> modelFilename{Name("modelFilename"), Comment("Path to RBDT .root model file")};
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
    _bdt(conf().modelKey(), ConfigFileLookupPolicy()(conf().modelFilename()))
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

      // extract track time at TT_Mid
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
        std::vector<float> features(_nFeatures);
        features[0] = cluster.GetAvgHitPos().x();
        features[1] = cluster.GetAvgHitPos().y();
        features[2] = cluster.GetAvgHitPos().z();
        features[3] = cluster.GetPEs();
        features[4] = trkTime - cluster.GetAvgHitTime();
        features[5] = cluster.GetCrvRecoPulses().size();
        features[6] = cluster.GetLayers().size();
        features[7] = cluster.GetSlope();
        features[8] = cluster.GetCrvSectorType();

        // run inference
        auto output = _bdt.Compute(features);
        float score = output[0];

        if (!ttMidFound) {
          score = 0;
        }

        if (_debug > 0) {
          printf("[CrvInference::%s] KalSeed %zu, CrvCoinc %zu: "
                 "features = [%.1f, %.1f, %.1f, %.1f, %.1f, %.0f, %.0f, %.3f, %d] "
                 "--> score = %.4f\n",
                 __func__, i_ks, i_crv,
                 features[0], features[1], features[2], features[3], features[4],
                 features[5], features[6], features[7], (int)features[8],
                 score);
        }

        auto crvCoincPtr = art::Ptr<CrvCoincidenceCluster>(crvCoincHandle, i_crv);
        assns->addSingle(kalSeedPtr, crvCoincPtr, MVAResult(score));
      }
    }

    event.put(std::move(assns));
  }

} // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvInference)
