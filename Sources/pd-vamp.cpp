#include <complex>
#include <m_pd.h>

#include <fftw3.h>
#include <rtvamp/hostsdk.hpp>

static t_class *VampObj;

using namespace rtvamp::hostsdk;
using FrequencyDomainBuffer = std::vector<const std::complex<float>>;
using TimeDomainBuffer = std::span<const float>;

// ─────────────────────────────────────
class PdVamp {
  public:
    t_object x_obj;
    t_sample Sample;

    // Vamp
    std::unique_ptr<Plugin> VampPlugin;
    Plugin::FeatureSet Features;
    Plugin::OutputList Outputs;

    bool Loaded;
    bool ReportErros = false;
    unsigned OutputCount = 0;

    // FFT
    float *FFTIn;
    fftwf_complex *FFTOut;
    fftwf_plan FFTPlan;

    // Audio
    std::vector<float> inBuffer;
    float Sr;
    int BlockSize;
    float HopSize;
    int BlockIndex;

    t_clock *Clock;
    t_outlet **Analisys;
    t_outlet *Info;
};

// ─────────────────────────────────────
static void GetParameterDescriptors(PdVamp *x) {
    size_t parameterIndex = 0;
    for (auto &&p : x->VampPlugin->getParameterDescriptors()) {
        post("<== Parameter %d ==>", parameterIndex);
        post("Identifier: %s", p.identifier.data());
        post("Name: %s", p.name.data());
        post("Description: %s", p.description.data());
        if (p.unit != "") {
            post("Unit: %s", p.unit.data());
        } else {
            post("Unit: N/A");
        }
        post("Default value: %f", p.defaultValue);
        post("Minimum value: %f", p.minValue);
        post("Maximum value: %f", p.maxValue);
        post("");
    }
}

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
                post("Plugin %s: %s", Id.c_str(), Name.c_str());
            } catch (const std::exception &e) {
                if (x->ReportErros) {
                    pd_error(nullptr, "Error loading plugin %s: %s", Id.c_str(), e.what());
                }
            }
        }
    }
}

// ─────────────────────────────────────
static void LoadVampPlugin(PdVamp *x, t_symbol *s) {
    std::string Id = s->s_name;
    x->VampPlugin = loadPlugin(Id, x->Sr);
    x->VampPlugin->initialise(x->HopSize, x->HopSize);
    x->OutputCount = x->VampPlugin->getOutputCount();
    x->Outputs = x->VampPlugin->getOutputDescriptors();
    if (x->OutputCount == 0) {
        pd_error(nullptr, "Plugin %s does not have any output", Id.c_str());
        return;
    }
    x->Loaded = true;
}

// ─────────────────────────────────────
static void VampTick(PdVamp *x) {
    for (unsigned i = 0; i < x->OutputCount; i++) {
        int Len = x->Features[i].size();
        if (Len == 0) {
            continue;
        } else if (Len == 1) {
            outlet_float(x->Analisys[i], x->Features[i][0]);
        } else {
            t_atom a[Len];
            for (int j = 0; j < Len; j++) {
                SETFLOAT(&a[j], x->Features[i][j]);
            }
            outlet_list(x->Analisys[i], nullptr, Len, a);
        }
    }

    return;
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

    if (x->VampPlugin->getInputDomain() == Plugin::InputDomain::Frequency) {
        for (int i = 0; i < x->BlockSize; i++) {
            x->FFTIn[i] *= 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (x->BlockSize - 1)));
        }
        fftwf_execute(x->FFTPlan);
        std::vector<std::complex<float>> FFT(x->HopSize);
        for (int i = 0; i < x->HopSize; i++) {
            FFT[i] = std::complex<float>(x->FFTOut[i][0], x->FFTOut[i][1]);
        }
        x->Features = x->VampPlugin->process(FFT, 0);

    } else {
        TimeDomainBuffer timeBuffer(x->inBuffer.data(), x->inBuffer.size());
        x->Features = x->VampPlugin->process(timeBuffer, 0);
    }

    x->BlockIndex = 0;
    clock_delay(x->Clock, 0);
    return (w + 4);
}

// ─────────────────────────────────────
static void PdVampAddDsp(PdVamp *x, t_signal **sp) {
    x->BlockIndex = 0;
    x->inBuffer.resize(x->HopSize, 0.0f);
    dsp_add(VampAudioPerform, 3, x, sp[0]->s_vec, sp[0]->s_n);
}

// ─────────────────────────────────────
static void PdVampFree(PdVamp *x, t_signal **sp) {
    fftwf_destroy_plan(x->FFTPlan);
    fftwf_free(x->FFTIn);
    fftwf_free(x->FFTOut);
}

// ─────────────────────────────────────
void *PdVampNew(t_symbol *s, int argc, t_atom *argv) {
    PdVamp *x = (PdVamp *)pd_new(VampObj);
    x->Sr = sys_getsr();
    x->Clock = clock_new(x, (t_method)VampTick);
    x->BlockIndex = 0;
    x->HopSize = 4096;
    x->Loaded = false;

    x->FFTIn = (float *)fftwf_alloc_real(x->HopSize);
    x->FFTOut = (fftwf_complex *)fftwf_alloc_complex(x->HopSize);
    x->FFTPlan = fftwf_plan_dft_r2c_1d(x->HopSize, x->FFTIn, x->FFTOut, FFTW_MEASURE);

    // check if 1 and 2 are symbols
    if (argc != 1) {
        pd_error(nullptr, "[vamp~] Wrong number of arguments: [vamp~ <plugin-id>]");
        post("Use plugins message to list available plugins");
        return x;
    }
    if (argv[0].a_type != A_SYMBOL) {
        pd_error(nullptr, "[vamp~] First argument must be symbols");
        return nullptr;
    }

    // Load Plugin
    std::string PluginId = atom_getsymbol(argv)->s_name;

    try {
        x->VampPlugin = loadPlugin(PluginId, x->Sr);
        x->VampPlugin->initialise(x->HopSize, x->HopSize);
        x->OutputCount = x->VampPlugin->getOutputCount();
        x->Outputs = x->VampPlugin->getOutputDescriptors();
        if (x->OutputCount == 0) {
            pd_error(nullptr, "Plugin %s does not have any output", PluginId.c_str());
            return nullptr;
        }
        x->Analisys = (t_outlet **)getbytes(x->OutputCount * sizeof(t_outlet *));
        for (unsigned i = 0; i < x->OutputCount; i++) {
            Plugin::OutputDescriptor p = x->Outputs[i];
            post("Output %d: %s", i, p.name.data());
            x->Analisys[i] = outlet_new(&x->x_obj, gensym(p.name.data()));
        }
    } catch (const std::exception &e) {
        pd_error(nullptr, "Error loading plugin %s: %s", PluginId.c_str(), e.what());
        return nullptr;
    }

    x->Loaded = true;

    return x;
}

// ─────────────────────────────────────
extern "C" void vamp_tilde_setup(void) {
    VampObj = class_new(gensym("vamp~"), (t_newmethod)PdVampNew, nullptr, sizeof(PdVamp),
                        CLASS_DEFAULT, A_GIMME, A_NULL);
    CLASS_MAINSIGNALIN(VampObj, PdVamp, Sample);
    class_addmethod(VampObj, (t_method)ListVampPlugins, gensym("plugins"), A_NULL);
    // class_addmethod(VampObj, (t_method)LoadVampPlugin, gensym("load"), A_SYMBOL, A_NULL);
    class_addmethod(VampObj, (t_method)GetParameterDescriptors, gensym("parameters"), A_NULL);
    // class_addmethod(VampObj, (t_method)GetOutputDescription, gensym("parameters"), A_NULL);
    class_addmethod(VampObj, (t_method)PdVampAddDsp, gensym("dsp"), A_CANT, 0);
}
