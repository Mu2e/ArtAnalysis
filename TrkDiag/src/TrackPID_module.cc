//
// Reconstruction-level PID determination from a track.  The current implementation
// just uses the calorimeter matching information (TrkCaloHit), eventually it will
// also process dE/dx, etc.
//
// Original author: Dave Brown (LBNL)
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
#include "Offline/ConfigTools/inc/ConfigFileLookupPolicy.hh"
// data
#include "Offline/RecoDataProducts/inc/KalSeed.hh"
#include "Offline/RecoDataProducts/inc/MVAResult.hh"
#include "ArtAnalysis/TrkDiag/inc/TrackPID.hxx"
#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "Offline/CalorimeterGeom/inc/DiskCalorimeter.hh"
//ONNX
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
// C++
#include <iostream>
#include <fstream>
#include <string>
#include <functional>
#include <float.h>
#include <vector>
using namespace std;

namespace TMVA_SOFIE_TrackPID {
  class Session;
}

namespace mu2e {
  class TrackPID : public art::EDProducer {
    public:
      struct Config {
        using Name=fhicl::Name;
        using Comment=fhicl::Comment;

        fhicl::Atom<float> MaxDE{Name("MaxDE"), Comment("Maximum difference between calorimeter cluster EDep energy and the track energy (assuming electron mass)")};
        fhicl::Atom<float> DT{ Name("DeltaTOffset"),
          Comment("Track - Calorimeter time offset")}; // this should be a condition FIXME!
        fhicl::Atom<art::InputTag> kalSeedPtrTag{Name("KalSeedPtrCollection"), Comment("Input tag for KalSeedPtrCollection")};
        fhicl::Atom<bool> printMVA{Name("printMVA"), Comment("print the MVA used"), false};
        fhicl::Atom<std::string> datFilename{Name("datFilename"), Comment("Filename for the .dat file to use")};
        fhicl::Atom<int> debug{Name("debugLevel"), Comment("Debug printout Level"), 0};
      };

      using Parameters = art::EDProducer::Table<Config>;
      TrackPID(const Parameters& conf);

    private:
      void produce(art::Event& event) override;
      void initializeMVA(std::string xmlfilename);

      float _maxde, _dtoffset;
      art::InputTag _kalSeedPtrTag;
      bool _printMVA;
      int _debug;

      std::shared_ptr<TMVA_SOFIE_TrackPID::Session> mva_;

      // Below is copied from TrackQuality_module.cc
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

      std::string print_shape(const std::vector<std::int64_t>& v) {
        std::stringstream ss("");
        for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
        ss << v[v.size() - 1];
        return ss.str();
      }
  };

  TrackPID::TrackPID(const Parameters& conf) :
    art::EDProducer(conf),
    _maxde(conf().MaxDE()),
    _dtoffset(conf().DT()),
    _kalSeedPtrTag(conf().kalSeedPtrTag()),
    _printMVA(conf().printMVA()),
    _debug(conf().debug()),
    _env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference"),
    _session(_env, "ArtAnalysis/TrkDiag/data/TrkQual_ANN1_v2.onnx", _session_options),
    _input_name(_session.GetInputNameAllocated(0, _allocator)),
    _type_info(_session.GetInputTypeInfo(0)),
    _tensor_info(_type_info.GetTensorTypeAndShapeInfo()),
    _input_shape(_tensor_info.GetShape()), // Get input shape from model
    _memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
    _output_name(_session.GetOutputNameAllocated(0, _allocator))
  {
    produces<MVAResultCollection>();

    ConfigFileLookupPolicy configFile;
    mva_ = std::make_shared<TMVA_SOFIE_TrackPID::Session>(configFile(conf().datFilename()));
    // Handle dynamic dimensions if needed
    for (auto& dim : _input_shape) {
      if (dim == -1) { dim = 1; }  // Set dynamic dims to 1 (or your desired value)
    }

      // Calculate total size
      _total_size = 1;
      for (auto dim : _input_shape) {
        _total_size *= dim;
      }
  }

  void TrackPID::produce(art::Event& event ) {
    mu2e::GeomHandle<mu2e::Calorimeter> calo;
    // create output
    unique_ptr<MVAResultCollection> mvacol(new MVAResultCollection());
    // get the KalSeedsPtrs
    art::Handle<KalSeedPtrCollection> kalSeedPtrHandle;
    event.getByLabel(_kalSeedPtrTag, kalSeedPtrHandle);
    const auto& kalSeedPtrs = *kalSeedPtrHandle;

    std::vector<float> input_tensor_values(_total_size, 0.0f);  // Initialize with zeros

    // Go through the tracks and calculate the track PID
    for (const auto& kalSeedPtr : kalSeedPtrs) {
      const auto& kalSeed = *kalSeedPtr;
      std::array<float,4> features{-9999,-9999,-9999,-9999}; // features used for training
      double mvaval = -1;

      // Fill the features
      static TrkFitFlag goodfit(TrkFitFlag::kalmanOK);
      if (kalSeed.status().hasAllProperties(goodfit)){
        if(kalSeed.hasCaloCluster() && kalSeed.caloHit()._flag.hasAllProperties(StrawHitFlag::active)){
          auto const& tchs = kalSeed.caloHit();
          auto const& cc = tchs.caloCluster();
          XYZVectorD trkmom = kalSeed.nearestSegment(tchs._rptoca)->momentum3();
          features[0] = cc->energyDep() - sqrt(trkmom.Mag2());
          // move into detector coordinates.  Yikes!!
          XYZVectorF cpos = XYZVectorF(calo->geomUtil().mu2eToTracker(calo->geomUtil().diskFFToMu2e( cc->diskID(), cc->cog3Vector())));
          features[1] = sqrt(cpos.Perp2());
          // compute transverse direction WRT position
          cpos.SetZ(0.0);
          trkmom.SetZ(0.0);
          features[2] = cpos.Dot(trkmom)/sqrt(cpos.Mag2()*trkmom.Mag2());
          // the following includes the (Calibrated) light-propagation time delay.  It should eventually be put in the reconstruction FIXME!
          // This velocity should come from conditions FIXME!
          features[3] = tchs.t0().t0()-tchs.time()- std::min((float)200.0,std::max((float)0.0,tchs.hitLen()))*0.005 - _dtoffset;

          // For ONNX:
          input_tensor_values[0] = features[0];
          input_tensor_values[1] = features[1];
          input_tensor_values[2] = features[2];
          input_tensor_values[3] = features[3]; 
          Ort::Value input_tensor = Ort::Value::CreateTensor<float>(_memory_info,
                                                                input_tensor_values.data(),
                                                                input_tensor_values.size(),
                                                                _input_shape.data(),
                                                                _input_shape.size()
                                                                );
          // Run inference
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
          float* output_data = output_tensors[0].GetTensorMutableData<float>();
              
          // hard cut on the energy difference.  This rejects cosmic rays which hit the calo and produce an upstream-going track that is then
          // reconstructed as a downstream particle associated to this cluster
          if(features[0] < _maxde){
            // evaluate the MVA
            auto mvaout = mva_->infer(features.data());
            mvaval = mvaout[0];
          } else {
             output_data[0] = 0; // for ONNX, set the output to 0 if the energy difference is above the threshold
          }

          if (_debug > 0) {
            printf("energy difference at %f, above threshold %f", features[0], _maxde);
          }
        }
      }
      if (_debug > 0) {
        printf("TrackPID ; input features: %f ; %f ; %f ; %f ; output: %f",
          features[0], features[1], features[2], features[3], mvaval);
      }
      mvacol->push_back(MVAResult(mvaval));
    }
    if (mvacol->size() != kalSeedPtrs.size()) {
      throw cet::exception("TrackPID") << "KalSeedPtr and MVAResult sizes are inconsistent: KalSeedPtr.size() = " << kalSeedPtrs.size() << " ; MVAResult.size() = " << mvacol->size();
    }
    // put the output products into the event
    event.put(move(mvacol));
  }
}// mu2e

DEFINE_ART_MODULE(mu2e::TrackPID)
