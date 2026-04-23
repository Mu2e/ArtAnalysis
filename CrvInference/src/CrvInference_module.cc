//
// Run XGBoost-based inference on CRV coincidence clusters
// paired with reconstructed tracks, producing an MVAResult
// score for each (KalSeed, CrvCoincidenceCluster) pair.
//
// Reference for XGBoost C API: 
// https://cms-ml.github.io/documentation/inference/xgboost.html
// 
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

// XGBoost
#include <xgboost/c_api.h>

// C++
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace mu2e {

  // type alias for associations
  using CrvInferenceAssns = art::Assns<KalSeed, CrvCoincidenceCluster, MVAResult>;

  class CrvInference : public art::EDProducer {

  public:
    struct Config {
      using Name = fhicl::Name;
      using Comment = fhicl::Comment;

      fhicl::Atom<art::InputTag> kalSeedPtrTag{Name("KalSeedPtrCollection"), Comment("Input tag for KalSeedPtrCollection")};
      fhicl::Atom<art::InputTag> crvCoincidenceTag{Name("CrvCoincidenceClusterCollection"), Comment("Input tag for CrvCoincidenceClusterCollection")};
      fhicl::Atom<std::string> modelFileName{Name("modelFileName"), Comment("Path to XGBoost .ubj model file")};
      fhicl::Atom<int> debug{Name("debugLevel"), Comment("Debug printout level"), 0};
    };

    using Parameters = art::EDProducer::Table<Config>;
    explicit CrvInference(const Parameters& conf);
    ~CrvInference();

  private:
    void produce(art::Event& event) override;

    art::InputTag _kalSeedPtrTag;
    art::InputTag _crvCoincidenceTag;
    int _debug;

    // Booster object
    BoosterHandle _booster;

    // Number of features is fixed, must match training!
    static constexpr size_t _nFeatures = 8;
  };

  CrvInference::CrvInference(const Parameters& conf) :
    art::EDProducer{conf},
    _kalSeedPtrTag(conf().kalSeedPtrTag()),
    _crvCoincidenceTag(conf().crvCoincidenceTag()),
    _debug(conf().debug())
  {
    produces<CrvInferenceAssns>();

    // Load XGBoost model
    if (XGBoosterCreate(nullptr, 0, &_booster) != 0)
      throw std::runtime_error(std::string("XGBoosterCreate failed: ") + XGBGetLastError());

    std::string modelPath = ConfigFileLookupPolicy()(conf().modelFileName());
    if (XGBoosterLoadModel(_booster, modelPath.c_str()) != 0)
      throw std::runtime_error(std::string("XGBoosterLoadModel failed: ") + XGBGetLastError());
  }

  CrvInference::~CrvInference() {
    XGBoosterFree(_booster);
  }

  void CrvInference::produce(art::Event& event) {

    auto assns = std::make_unique<CrvInferenceAssns>();

    const auto& kalSeedPtrHandle = event.getValidHandle<KalSeedPtrCollection>(_kalSeedPtrTag);
    const auto& crvCoincHandle = event.getValidHandle<CrvCoincidenceClusterCollection>(_crvCoincidenceTag);

    // Iterate through kalseeds
    for (size_t i_ks = 0; i_ks < kalSeedPtrHandle->size(); ++i_ks) {
      const auto& kalSeedPtr = kalSeedPtrHandle->at(i_ks);
      const auto& kalSeed = *kalSeedPtr;

      double trkTime = -9999;
      bool ttMidFound = false;
      for (const auto& kinter : kalSeed.intersections()) {
        if (kinter.surfaceId() == SurfaceIdDetail::TT_Mid) {
          trkTime = kinter.time();
          ttMidFound = true;
          break;
        }
      }
      
      // iterate through coincidence clusters
      for (size_t i_crv = 0; i_crv < crvCoincHandle->size(); ++i_crv) {
        const auto& cluster = crvCoincHandle->at(i_crv);

        // build feature vector (order must match training!)
        std::vector<float> features(_nFeatures);
        features[0] = cluster.GetAvgHitPos().x();
        features[1] = cluster.GetAvgHitPos().y();
        features[2] = cluster.GetAvgHitPos().z();
        features[3] = cluster.GetPEs();
        features[4] = trkTime - cluster.GetAvgHitTime();
        features[5] = cluster.GetCrvRecoPulses().size();
        features[6] = cluster.GetLayers().size();
        features[7] = cluster.GetSlope();

        // run XGBoost inference
        // See reference in preamble comments...
        float score = 0.0f;
        if (ttMidFound) {
          DMatrixHandle dmat;
          if (XGDMatrixCreateFromMat(features.data(), 1, _nFeatures, NAN, &dmat) != 0) {
            throw std::runtime_error(std::string("XGDMatrixCreateFromMat failed: ") + XGBGetLastError());
          }

          bst_ulong out_len = 0;
          const float* out_result = nullptr;
          if (XGBoosterPredict(_booster, dmat, 0, 0, 0, &out_len, &out_result) != 0) {
            XGDMatrixFree(dmat);
            throw std::runtime_error(std::string("XGBoosterPredict failed: ") + XGBGetLastError());
          }
          if (out_len < 1 || out_result == nullptr) {
            XGDMatrixFree(dmat);
            throw std::runtime_error("XGBoosterPredict returned no result");
          }

          score = out_result[0];
          if (XGDMatrixFree(dmat) != 0) {
            throw std::runtime_error(std::string("XGDMatrixFree failed: ") + XGBGetLastError());
          }
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
                    << ", dt = " << features[4] 
                    << ", nHits = " << features[5] 
                    << ", nLayers = " << features[6]
                    << ",  slope = " << features[7] << std::endl
                    << "  score = " << score << std::endl;
        }

        auto crvCoincPtr = art::Ptr<CrvCoincidenceCluster>(crvCoincHandle, i_crv);
        assns->addSingle(kalSeedPtr, crvCoincPtr, MVAResult(score));
      }
    }

    // Add the association to the art::Event
    event.put(std::move(assns));
  }

} // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvInference)