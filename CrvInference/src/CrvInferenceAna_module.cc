//
// analyzer to histogram CrvInference BDT scores.
// Reads the art::Assns<KalSeed, CrvCoincidenceCluster, MVAResult>
// produced by CrvInference and fills a TH1F with scores
//
// Sam Grant 2026

// art
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art_root_io/TFileService.h"
#include "fhiclcpp/types/Atom.h"
#include "canvas/Persistency/Common/Assns.h"

// Offline
#include "Offline/RecoDataProducts/inc/KalSeed.hh"
#include "Offline/RecoDataProducts/inc/CrvCoincidenceCluster.hh"
#include "Offline/RecoDataProducts/inc/MVAResult.hh"

// ROOT
#include "TH1F.h"

namespace mu2e {

  using CrvInferenceAssns = art::Assns<KalSeed, CrvCoincidenceCluster, MVAResult>;

  class CrvInferenceAna : public art::EDAnalyzer {

  public:
    struct Config {
      using Name = fhicl::Name;
      using Comment = fhicl::Comment;

      fhicl::Atom<art::InputTag> crvInferenceTag{Name("CrvInferenceTag"), Comment("Input tag for CrvInference associations")};
    };

    using Parameters = art::EDAnalyzer::Table<Config>;
    explicit CrvInferenceAna(const Parameters& conf);

  private:
    void analyze(const art::Event& event) override;

    art::InputTag _crvInferenceTag;

    TH1F* _hScore;
  };

  CrvInferenceAna::CrvInferenceAna(const Parameters& conf) :
    art::EDAnalyzer{conf},
    _crvInferenceTag(conf().crvInferenceTag())
  {
    art::ServiceHandle<art::TFileService> tfs;
    _hScore = tfs->make<TH1F>("h1_score", "CrvInference BDT score;Score;Entries", 100, 0.0, 1.0);
  }

  void CrvInferenceAna::analyze(const art::Event& event) {
    const auto& assns = event.getValidHandle<CrvInferenceAssns>(_crvInferenceTag);

    for (const auto& assn : *assns) {
      float score = assn.data->_value;
      _hScore->Fill(score);
    }
  }

} // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvInferenceAna)