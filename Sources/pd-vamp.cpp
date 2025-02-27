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
    bool FrequencyDomain = false;
    unsigned OutputCount = 0;

    // FFT
    float *FFTIn;
    fftwf_complex *FFTOut;
    fftwf_plan FFTPlan;

    // Audio
    std::vector<float> inBuffer;
    float Sr;
    int BlockSize;
    float StepSize;
    int BlockIndex;

    t_clock *Clock;
    t_outlet *Analisys;
    t_outlet *Info;
};

// ─────────────────────────────────────
static void vamp_getparameters(PdVamp *x) {
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
        parameterIndex++;
    }
    if (parameterIndex == 0) {
        post("[vamp~] No parameters available!");
    }
}

// ─────────────────────────────────────
static void vamp_getprograms(PdVamp *x) {
    size_t programIndex = 0;
    for (auto &&p : x->VampPlugin->getPrograms()) {
        post("<== Program %d ==>", programIndex);
        post("Name: %s", p.data());
        programIndex += 1;
    }
    if (x->VampPlugin->getCurrentProgram()) {
        post("Current Program: %s", x->VampPlugin->getCurrentProgram()->data());
    }
    if (programIndex == 0) {
        post("[vamp~] No programs available!");
    }
}

// ─────────────────────────────────────
static void vamp_listplugins(PdVamp *x) {
    for (auto &&lib : listLibraries()) {
        for (auto &&key : listPlugins(lib)) {
            std::string Id = key.get().data();
            try {
                const auto Plugin = loadPlugin(key, 48000);
                std::string Name = Plugin->getName().data();
                std::string Description = Plugin->getDescription().data();
                post("'%s': %s | %s", Id.c_str(), Name.c_str(), Description.c_str());
            } catch (const std::exception &e) {
                if (x->ReportErros) {
                    pd_error(nullptr, "Error loading plugin %s: %s", Id.c_str(), e.what());
                }
            }
        }
    }
}

// ─────────────────────────────────────
static void VampTick(PdVamp *x) {
    if (x->OutputCount == 0) {
        return;
    } else {
        int Len = x->Features[0].size();
        if (Len == 0) {
            return;
        } else if (Len == 1) {
            outlet_float(x->Analisys, x->Features[0][0]);
        } else {
            t_atom a[Len];
            for (int j = 0; j < Len; j++) {
                SETFLOAT(&a[j], x->Features[0][j]);
            }
            outlet_list(x->Analisys, nullptr, Len, a);
        }
    }

    return;
}

// ─────────────────────────────────────
static t_int *vamp_perform(t_int *w) { //
    PdVamp *x = (PdVamp *)(w[1]);
    t_sample *in = (t_sample *)(w[2]);
    int n = static_cast<int>(w[3]);

    if (!x->Loaded) {
        return (w + 4);
    }

    x->BlockIndex += n;
    std::copy(x->inBuffer.begin() + n, x->inBuffer.end(), x->inBuffer.begin());
    std::copy(in, in + n, x->inBuffer.end() - n);

    if (x->BlockIndex != x->StepSize) {
        return (w + 4);
    }

    if (x->FrequencyDomain) {
        for (int i = 0; i < x->BlockSize; i++) {
            x->FFTIn[i] *= 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (x->BlockSize - 1)));
        }
        fftwf_execute(x->FFTPlan);
        std::vector<std::complex<float>> FFT(x->StepSize);
        for (int i = 0; i < x->StepSize; i++) {
            FFT[i] = std::complex<float>(x->FFTOut[i][0], x->FFTOut[i][1]);
        }
        x->Features = x->VampPlugin->process(FFT, 0);

    } else {
        TimeDomainBuffer TimeBuffer(x->inBuffer.data(), x->inBuffer.size());
        x->Features = x->VampPlugin->process(TimeBuffer, 0);
    }

    x->BlockIndex = 0;
    clock_delay(x->Clock, 0);
    return (w + 4);
}

// ─────────────────────────────────────
static void vamp_dsp(PdVamp *x, t_signal **sp) {
    x->BlockIndex = 0;
    x->inBuffer.resize(x->StepSize, 0.0f);
    dsp_add(vamp_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
}

// ─────────────────────────────────────
static void PdVampFree(PdVamp *x, t_signal **sp) {
    fftwf_destroy_plan(x->FFTPlan);
    fftwf_free(x->FFTIn);
    fftwf_free(x->FFTOut);
}

// ─────────────────────────────────────
void *vamp_new(t_symbol *s, int argc, t_atom *argv) {
    PdVamp *x = (PdVamp *)pd_new(VampObj);
    x->Sr = sys_getsr();
    x->Clock = clock_new(x, (t_method)VampTick);
    x->BlockIndex = 0;
    x->StepSize = 1024;
    x->BlockSize = 1024;
    x->Loaded = false;

    x->FFTIn = (float *)fftwf_alloc_real(x->StepSize);
    x->FFTOut = (fftwf_complex *)fftwf_alloc_complex(x->StepSize);
    x->FFTPlan = fftwf_plan_dft_r2c_1d(x->StepSize, x->FFTIn, x->FFTOut, FFTW_MEASURE);

    // check if 1 and 2 are symbols
    if (argc != 1) {
        pd_error(nullptr, "[vamp~] Wrong number of arguments: [vamp~ <plugin-id>]");
        post("Use [plugins] method to list available plugins");
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
        x->VampPlugin->initialise(x->StepSize, x->BlockSize);
        x->Loaded = true;
        if (x->VampPlugin->getInputDomain() == Plugin::InputDomain::Frequency) {
            x->FrequencyDomain = true;
        } else {
            x->FrequencyDomain = false;
        }
    } catch (const std::exception &e) {
        x->Loaded = false;
        pd_error(nullptr, "Error loading plugin %s: %s", PluginId.c_str(), e.what());
        return nullptr;
    }
    x->OutputCount = x->VampPlugin->getOutputCount();
    x->Outputs = x->VampPlugin->getOutputDescriptors();
    if (x->OutputCount == 0) {
        pd_error(nullptr, "Plugin %s does not have any output", PluginId.c_str());
        return nullptr;
    }
    x->Analisys = outlet_new(&x->x_obj, &s_anything);
    return x;
}

// ─────────────────────────────────────
extern "C" void vamp_tilde_setup(void) {
    VampObj = class_new(gensym("vamp~"), (t_newmethod)vamp_new, nullptr, sizeof(PdVamp),
                        CLASS_DEFAULT, A_GIMME, A_NULL);
    CLASS_MAINSIGNALIN(VampObj, PdVamp, Sample);
    class_addmethod(VampObj, (t_method)vamp_listplugins, gensym("plugins"), A_NULL);
    class_addmethod(VampObj, (t_method)vamp_getparameters, gensym("parameters"), A_NULL);
    class_addmethod(VampObj, (t_method)vamp_getprograms, gensym("program"), A_NULL);
    class_addmethod(VampObj, (t_method)vamp_dsp, gensym("dsp"), A_CANT, 0);
}
