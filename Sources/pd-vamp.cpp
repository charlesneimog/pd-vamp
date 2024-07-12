#include <m_pd.h>

#include <cmath>
#include <complex>
#include <fftw3.h>
#include <span>
#include <vector>

#include <fftw3.h>
#include <rtvamp/hostsdk.hpp>

static t_class *VampObj;

using namespace rtvamp::hostsdk;

// ─────────────────────────────────────
class PdVamp {
  public:
    t_object x_obj;
    t_sample Sample;
    bool ReportErros = false;
    std::unique_ptr<Plugin> VampPlugin;
    bool Loaded = false;

    // FFT
    float *FFTIn;
    fftwf_complex *FFTOut;
    fftwf_plan FFTPlan;

    std::vector<float> inBuffer;
    float Sr;
    int BlockSize;
    float HopSize;
    int BlockIndex;

    t_clock *Clock;

    t_outlet *analysis_outlet;
    t_outlet *info;
};

// ─────────────────────────────────────
static void LoadVampPlugin(PdVamp *x, t_symbol *s) {
    std::string Id = s->s_name;
    x->VampPlugin = loadPlugin(Id, x->Sr);
    if (x->VampPlugin->getInputDomain() == Plugin::InputDomain::Frequency) {
        pd_error(nullptr, "Plugin %s is time domain | not implemented yet",
                 Id.c_str());
    }
    x->VampPlugin->initialise(x->BlockSize, x->BlockSize);
    x->Loaded = true;
}

// ─────────────────────────────────────
static void VampTick(PdVamp *x) { post("ok"); }

// ─────────────────────────────────────
static void ListVampPlugins(PdVamp *x) {
    for (auto &&lib : listLibraries()) {
        for (auto &&key : listPlugins(lib)) {
            std::string Id = key.get().data();
            try {
                const auto Plugin = loadPlugin(key, 48000);
                std::string Name = Plugin->getName().data();
                t_atom a[2];
                SETSYMBOL(&a[0], gensym(Id.c_str()));
                SETSYMBOL(&a[1], gensym(Name.c_str()));
                outlet_anything(x->info, gensym("plugin"), 2, a);

            } catch (const std::exception &e) {
                if (x->ReportErros) {
                    pd_error(nullptr, "Error loading plugin %s: %s", Id.c_str(),
                             e.what());
                }
            }
        }
    }
}

// ─────────────────────────────────────
static t_int *VampAudioPerform(t_int *w) { //
    PdVamp *x = (PdVamp *)(w[1]);
    t_sample *in = (t_sample *)(w[2]);
    int n = static_cast<int>(w[3]);

    if (!x->Loaded) {
        return (w + 4);
    }

    x->BlockIndex += n;
    std::copy(x->inBuffer.begin() + n, x->inBuffer.end(), x->inBuffer.begin());
    std::copy(in, in + n, x->inBuffer.end() - n);

    if (x->BlockIndex != x->HopSize) {
        return (w + 4);
    }

    std::span<const std::complex<float>>
        bufferChannel; // Define this appropriately

    const auto getInputBuffer = [&]() -> Plugin::InputBuffer {
        if (x->VampPlugin->getInputDomain() == Plugin::InputDomain::Frequency) {
            for (int i = 0; i < x->BlockSize; i++) {
                x->FFTIn[i] *=
                    0.5 * (1.0 - std::cos(2.0 * M_PI * i / (x->BlockSize - 1)));
            }
            fftwf_execute(x->FFTPlan);

            // Convert FFTW output to std::span<const std::complex<float>>
            return std::span<const std::complex<float>>(
                reinterpret_cast<const std::complex<float> *>(x->FFTOut),
                x->BlockSize);
        } else {
            // Assuming bufferChannel is defined elsewhere and holds the
            // appropriate data
            return bufferChannel;
        }
    };

    x->BlockIndex = 0;
    auto features = x->VampPlugin->process(getInputBuffer(), 0);
    return (w + 4);
}

// ─────────────────────────────────────
static void VampAddDsp(PdVamp *x, t_signal **sp) {
    x->BlockIndex = 0;
    x->inBuffer.resize(x->HopSize, 0.0f);

    // TODO: Free
    x->FFTIn = (float *)fftwf_alloc_real(x->BlockSize);
    x->FFTOut = (fftwf_complex *)fftwf_alloc_complex(x->BlockSize);
    x->FFTPlan =
        fftwf_plan_dft_r2c_1d(x->BlockSize, x->FFTIn, x->FFTOut, FFTW_MEASURE);

    dsp_add(VampAudioPerform, 3, x, sp[0]->s_vec, sp[0]->s_n);
}

// ─────────────────────────────────────
void *PdVampNew(void) {
    PdVamp *x = (PdVamp *)pd_new(VampObj);
    x->analysis_outlet = outlet_new(&x->x_obj, &s_anything);
    x->info = outlet_new(&x->x_obj, &s_anything);
    x->Sr = sys_getsr();
    x->Clock = clock_new(x, (t_method)VampTick);
    x->BlockIndex = 0;
    x->HopSize = 4096;
    x->Loaded = false;
    return x;
}

// ─────────────────────────────────────
extern "C" void vamp_tilde_setup(void) {
    VampObj = class_new(gensym("vamp~"), (t_newmethod)PdVampNew, nullptr,
                        sizeof(PdVamp), CLASS_DEFAULT, A_NULL);
    CLASS_MAINSIGNALIN(VampObj, PdVamp, Sample);
    class_addmethod(VampObj, (t_method)ListVampPlugins, gensym("plugins"),
                    A_NULL);
    class_addmethod(VampObj, (t_method)LoadVampPlugin, gensym("load"), A_SYMBOL,
                    A_NULL);
    class_addmethod(VampObj, (t_method)VampAddDsp, gensym("dsp"), A_CANT, 0);
}
