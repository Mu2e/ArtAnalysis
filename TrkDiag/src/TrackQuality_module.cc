//
// Create a TrkQual object
// using TMVA::SOFIE
//
// Original author A. Edmonds
//

// framework
#include "art/Framework/Principal/Event.h"
#include "fhiclcpp/ParameterSet.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Core/EDProducer.h"
#include "art_root_io/TFileService.h"
#include "art/Utilities/make_tool.h"
// utilities
#include "Offline/ProditionsService/inc/ProditionsHandle.hh"
#include "Offline/Mu2eUtilities/inc/MVATools.hh"
#include "Offline/GlobalConstantsService/inc/GlobalConstantsHandle.hh"
#include "Offline/GlobalConstantsService/inc/ParticleDataList.hh"
#include "Offline/ConfigTools/inc/ConfigFileLookupPolicy.hh"
// data
#include "Offline/RecoDataProducts/inc/KalSeed.hh"
#include "Offline/RecoDataProducts/inc/MVAResult.hh"
// ONNXRuntime
#include "onnxruntime/onnxruntime_cxx_api.h"
// XGBoost
#include <xgboost/c_api.h>
// C++
#include <iostream>
#include <fstream>
#include <string>
#include <functional>
#include <float.h>
#include <vector>
using namespace std;
using CLHEP::Hep3Vector;
using CLHEP::HepVector;

namespace mu2e
{

  class TrackQuality : public art::EDProducer
  {
    public:
      struct Config {
        using Name=fhicl::Name;
        using Comment=fhicl::Comment;

        fhicl::Atom<art::InputTag> kalSeedPtrTag{Name("KalSeedPtrCollection"), Comment("Input tag for KalSeedPtrCollection")};
        fhicl::Atom<bool> printMVA{Name("PrintMVA"), Comment("Print the MVA used"), false};
        fhicl::Atom<std::string> onnxFilename{Name("onnxFilename"), Comment("Filename for the .onnx file to use")};
        fhicl::Atom<std::string> xgbFileName{Name("xgbFilename"), Comment("Path to XGBoost .ubj model file")};
        fhicl::Atom<int> debug{Name("debugLevel"), Comment("Debug printout level"), 0};
      };

      using Parameters = art::EDProducer::Table<Config>;
      TrackQuality(const Parameters& conf);

    private:
      void produce(art::Event& event) override;
      void initializeMVA(std::string xmlfilename);

      art::InputTag _kalSeedPtrTag;
      bool _printMVA;
      int _debug;

    ConfigFileLookupPolicy _configFileLookup;

    // for the ANN
    Ort::Env _env;
    Ort::SessionOptions _session_options;
    Ort::Session _session;
    Ort::AllocatorWithDefaultOptions _allocator;
    Ort::AllocatedStringPtr _input_name;
    Ort::TypeInfo _type_info;
    Ort::ConstTensorTypeAndShapeInfo _tensor_info;
    std::vector<int64_t> _input_shape;
    size_t _total_size;
    Ort::MemoryInfo _memory_info;
    Ort::AllocatedStringPtr _output_name;

    // for the BDT
    BoosterHandle _booster; // Booster object
    // Number of features is fixed, must match training!
    static constexpr size_t _nFeatures = 7;
  };

  TrackQuality::TrackQuality(const Parameters& conf) :
    art::EDProducer{conf},
    _kalSeedPtrTag(conf().kalSeedPtrTag()),
    _printMVA(conf().printMVA()),
    _debug(conf().debug()),
    // Note that the following is required to be in order
    _env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference"),
    _session(_env, _configFileLookup(conf().onnxFilename()).c_str(), _session_options),
    _input_name(_session.GetInputNameAllocated(0, _allocator)),
    _type_info(_session.GetInputTypeInfo(0)),
    _tensor_info(_type_info.GetTensorTypeAndShapeInfo()),
    _input_shape(_tensor_info.GetShape()), // Get input shape from model
    _memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
    _output_name(_session.GetOutputNameAllocated(0, _allocator))
    {
      produces<MVAResultCollection>("ANN");
      produces<MVAResultCollection>("BDT");

      // Handle dynamic dimensions if needed
      for (auto& dim : _input_shape) {
        if (dim == -1) { dim = 1; }  // Set dynamic dims to 1 (or your desired value) TODO: can set to any number so we can batch N tracks per event
      }

      // Calculate total size
      _total_size = 1;
      for (auto dim : _input_shape) {
        _total_size *= dim;
      }

      // Load XGBoost model
      if (XGBoosterCreate(nullptr, 0, &_booster) != 0) {
        throw std::runtime_error(std::string("XGBoosterCreate failed: ") + XGBGetLastError());
      }

      std::string modelPath = ConfigFileLookupPolicy()(conf().xgbFileName());
      if (XGBoosterLoadModel(_booster, modelPath.c_str()) != 0) {
        throw std::runtime_error(std::string("XGBoosterLoadModel failed: ") + XGBGetLastError());
      }

      // verify the loaded model matches the expected feature count
      bst_ulong nFeaturesModel = 0;
      if (XGBoosterGetNumFeature(_booster, &nFeaturesModel) != 0) {
        throw std::runtime_error(std::string("XGBoosterGetNumFeature failed: ") + XGBGetLastError());
      }

    }

  void TrackQuality::produce(art::Event& event ) {
    // create output
    unique_ptr<MVAResultCollection> anncol(new MVAResultCollection());
    unique_ptr<MVAResultCollection> bdtcol(new MVAResultCollection());

    // get the KalSeedPtrs
    art::Handle<KalSeedPtrCollection> kalSeedPtrHandle;
    event.getByLabel(_kalSeedPtrTag, kalSeedPtrHandle);
    const auto& kalSeedPtrs = *kalSeedPtrHandle;

    // Prepare input tensor
    std::vector<float> features(_total_size, 0.0f);  // Initialize with zeros


    // Go through the tracks and calculate their track qualities
    for (const auto& kalSeedPtr : kalSeedPtrs) {
      const auto& kalSeed = *kalSeedPtr;

      // fill the hit count variables
      int nhits = 0; int nactive = 0; int ndouble = 0; int ndactive = 0; int nnullambig = 0;
      static StrawHitFlag active(StrawHitFlag::active);
        if(_debug > 1) printf("[TrackQuality::%s::%s] Printing track hit information\n", __func__, moduleDescription().moduleLabel().c_str());
      for (auto ihit = kalSeed.hits().begin(); ihit != kalSeed.hits().end(); ++ihit) {
        ++nhits;
        if (ihit->flag().hasAllProperties(active)) {
          ++nactive;
          if (ihit->ambig()==0) {
            ++nnullambig;
          }
        }
        if(_debug > 1) printf(" Hit %2i: active = %o, null = %o\n",
                              nhits, ihit->flag().hasAllProperties(active), ihit->flag().hasAllProperties(active) && ihit->ambig()==0);
        auto jhit = ihit; jhit++;
        if(jhit != kalSeed.hits().end() && ihit->strawId().uniquePanel() ==
           jhit->strawId().uniquePanel()){
          ++ndouble;
          if(ihit->flag().hasAllProperties(active)) { ++ndactive; }
        }
      }

      int ndof = nactive -5;
      if (kalSeed.hasCaloCluster()) {
        ++ndof;
      }

      int nmat = 0; int nmatactive = 0; int radlen = 0.0;
      for (std::vector<TrkStraw>::const_iterator i_straw = kalSeed.straws().begin(); i_straw != kalSeed.straws().end(); ++i_straw) {
        ++nmat;
        if (i_straw->active()) {
          ++nmatactive;
          radlen += i_straw->_radlen;
        }
      }

      features[0] = nactive;
      features[1] = (double) nactive / nhits;
      features[3] = (double) nnullambig / nactive;
      features[4] = kalSeed.fitConsistency();
      features[6] = (double)nmatactive / nactive;

      // Now get the features that are for the entrance of the trackre
      bool entrance_found = false;
      for(size_t ikinter = 0; ikinter < kalSeed.intersections().size(); ++ikinter){
        auto const& kinter = kalSeed.intersections()[ikinter];
        if (kinter.surfaceId() == SurfaceIdDetail::TT_Front) { // we only want the tracker entrance (sid=0)
          features[2] = sqrt(kinter.loopHelix().paramVar(KinKal::LoopHelix::t0_));
          features[5] = kinter.momerr();
          entrance_found = true;
          break;
        }
      }
      if (!entrance_found) {
        features[2] = -9999;
        features[5] = -9999;
      }

      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(_memory_info,
                                                                features.data(),
                                                                features.size(),
                                                                _input_shape.data(),
                                                                _input_shape.size()
                                                                );
      // Run ANN inference
      const char* input_names[] = {_input_name.get()};
      const char* output_names[] = {_output_name.get()};
      auto output_tensors = _session.Run(
                                         Ort::RunOptions{nullptr},
                                         input_names,
                                         &input_tensor,
                                         1,
                                         output_names,
                                         1
                                         );
      // Get output
      float* annout = output_tensors[0].GetTensorMutableData<float>();

      if (!entrance_found) {
        annout[0] = 0; // this is not a good track
      }

      // run XGBoost inference
      // See reference in preamble comments...
      float bdt_score = 0.0f;
      DMatrixHandle dmat;
      if (XGDMatrixCreateFromMat(features.data(), 1, _nFeatures, NAN, &dmat) != 0) { // use same features vector
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

      bdt_score = out_result[0];
      if (XGDMatrixFree(dmat) != 0) {
        throw std::runtime_error(std::string("XGDMatrixFree failed: ") + XGBGetLastError());
      }


      if(_debug > 0) {
        printf("[TrackQuality::%s::%s] Inputs = %.0f, %.4f, %.4f, %.4f, %.4f, %.4f %.4f --> ANN output = %.4fm BDT output = %.4fm\n",
               __func__, moduleDescription().moduleLabel().c_str(),
               features[0], features[1], features[2], features[3], features[4], features[5], features[6], annout[0], bdt_score);
      }

      anncol->push_back(MVAResult(annout[0]));
      bdtcol->push_back(MVAResult(bdt_score));
    }

    if ( (anncol->size() != kalSeedPtrs.size()) ) {
      throw cet::exception("TrackQuality") << "KalSeedPtr and MVAResult sizes are inconsistent (" << kalSeedPtrs.size() << ", " << anncol->size();
    }


    // put the output products into the event
    event.put(move(anncol), "ANN");
    event.put(move(bdtcol), "BDT");
  }
}// mu2e

DEFINE_ART_MODULE(mu2e::TrackQuality)
