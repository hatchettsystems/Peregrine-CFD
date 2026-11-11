// C ABI around OpenCASCADE: read a STEP assembly with names and transforms, return one triangle
// mesh per leaf body (millimetres). Also a tiny writer used by the tests.
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <BRepAlgoAPI_Cut.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRep_Tool.hxx>
#include <Interface_Static.hxx>
#include <Poly_Triangulation.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <STEPControl_StepModelType.hxx>
#include <TCollection_AsciiString.hxx>
#include <TDF_Label.hxx>
#include <TDF_LabelSequence.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Compound.hxx>
#include <BRep_Builder.hxx>
#include <StepRepr_RepresentationItem.hxx>
#include <TCollection_HAsciiString.hxx>
#include <XSControl_TransferReader.hxx>
#include <XSControl_WorkSession.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <gp_Trsf.hxx>

namespace {
struct Body {
    std::string name;
    std::vector<double> verts;
    std::vector<uint32_t> tris;
};
}

struct PgStep {
    std::vector<Body> bodies;
};

static std::string label_name(const TDF_Label& l) {
    Handle(TDataStd_Name) n;
    if (l.FindAttribute(TDataStd_Name::GetID(), n)) {
        TCollection_AsciiString a(n->Get(), '?');
        return std::string(a.ToCString());
    }
    return std::string();
}

static void set_err(char* err, int len, const std::string& m) {
    if (err && len > 0) {
        std::strncpy(err, m.c_str(), len - 1);
        err[len - 1] = 0;
    }
}

static void mesh_into(const TopoDS_Shape& s, double deflection, Body& b) {
    BRepMesh_IncrementalMesh mesher(s, deflection, Standard_False, 0.3, Standard_True);
    for (TopExp_Explorer ex(s, TopAbs_FACE); ex.More(); ex.Next()) {
        TopoDS_Face face = TopoDS::Face(ex.Current());
        TopLoc_Location loc;
        Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(face, loc);
        if (tri.IsNull()) continue;
        const bool reversed = face.Orientation() == TopAbs_REVERSED;
        const uint32_t off = (uint32_t)(b.verts.size() / 3);
        const gp_Trsf t = loc.Transformation();
        for (int i = 1; i <= tri->NbNodes(); ++i) {
            gp_Pnt p = tri->Node(i).Transformed(t);
            b.verts.push_back(p.X());
            b.verts.push_back(p.Y());
            b.verts.push_back(p.Z());
        }
        for (int i = 1; i <= tri->NbTriangles(); ++i) {
            int a, c, d;
            tri->Triangle(i).Get(a, c, d);
            if (reversed) std::swap(c, d);
            b.tris.push_back(off + a - 1);
            b.tris.push_back(off + c - 1);
            b.tris.push_back(off + d - 1);
        }
    }
}

// Name stored on the STEP entity (e.g. MANIFOLD_SOLID_BREP('Body3', ...)) that produced `s`.
static std::string step_item_name(const Handle(XSControl_TransferReader)& tr, const TopoDS_Shape& s) {
    if (tr.IsNull() || s.IsNull()) return std::string();
    try {
        Handle(Standard_Transient) ent = tr->EntityFromShapeResult(s, 1);
        Handle(StepRepr_RepresentationItem) item = Handle(StepRepr_RepresentationItem)::DownCast(ent);
        if (!item.IsNull() && !item->Name().IsNull()) return std::string(item->Name()->ToCString());
    } catch (...) {
    }
    return std::string();
}

static void walk(const Handle(XCAFDoc_ShapeTool)& st, const Handle(XSControl_TransferReader)& tr, const TDF_Label& label, const TopLoc_Location& parent, double defl, PgStep* out, int depth) {
    if (depth > 64) return;
    TDF_Label ref = label;
    if (st->IsReference(label)) st->GetReferredShape(label, ref);
    TopLoc_Location loc = parent * XCAFDoc_ShapeTool::GetLocation(label);
    if (st->IsAssembly(ref)) {
        TDF_LabelSequence comps;
        st->GetComponents(ref, comps);
        for (int i = 1; i <= comps.Length(); ++i) walk(st, tr, comps.Value(i), loc, defl, out, depth + 1);
        return;
    }
    TopoDS_Shape shape = st->GetShape(ref);
    if (shape.IsNull()) return;
    const std::string product = [&] {
        std::string n = label_name(ref);
        if (n.empty()) n = label_name(label);
        return n.empty() ? std::string("body") : n;
    }();
    // A product may hold several solids (e.g. a toolhead exported as one part with Body1, Body2, ...).
    // Return each solid as its own body so the user can decide which ones form the duct.
    std::vector<std::pair<std::string, TopoDS_Shape>> parts;
    TDF_LabelSequence subs;
    st->GetSubShapes(ref, subs);
    for (int i = 1; i <= subs.Length(); ++i) {
        TopoDS_Shape sub = st->GetShape(subs.Value(i));
        if (sub.IsNull() || sub.ShapeType() != TopAbs_SOLID) continue;
        std::string n = label_name(subs.Value(i));
        parts.emplace_back(n, sub);
    }
    if (parts.size() < 2) {
        parts.clear();
        int k = 0;
        for (TopExp_Explorer ex(shape, TopAbs_SOLID); ex.More(); ex.Next()) {
            parts.emplace_back(std::string(), ex.Current());
            ++k;
        }
        if (k < 2) parts.clear();
    }
    if (parts.size() < 2) {
        Body b;
        b.name = product;
        mesh_into(shape.Moved(loc), defl, b);
        if (!b.tris.empty()) out->bodies.push_back(std::move(b));
        return;
    }
    int idx = 0;
    for (auto& pr : parts) {
        ++idx;
        Body b;
        std::string nm = pr.first;
        if (nm.empty()) nm = step_item_name(tr, pr.second);
        b.name = nm.empty() ? product + " " + std::to_string(idx) : nm;
        mesh_into(pr.second.Moved(loc), defl, b);
        if (!b.tris.empty()) out->bodies.push_back(std::move(b));
    }
}

extern "C" {

PgStep* pg_step_read(const char* path, double deflection, char* err, int err_len) {
    try {
        Interface_Static::SetCVal("xstep.cascade.unit", "MM");
        Handle(XCAFApp_Application) app = XCAFApp_Application::GetApplication();
        Handle(TDocStd_Document) doc;
        app->NewDocument("MDTV-XCAF", doc);
        STEPCAFControl_Reader reader;
        reader.SetNameMode(Standard_True);
        reader.SetColorMode(Standard_False);
        reader.SetLayerMode(Standard_False);
        reader.SetPropsMode(Standard_False);
        if (reader.ReadFile(path) != IFSelect_RetDone) {
            set_err(err, err_len, "cannot read STEP file");
            return nullptr;
        }
        if (!reader.Transfer(doc)) {
            set_err(err, err_len, "STEP transfer failed");
            return nullptr;
        }
        Handle(XCAFDoc_ShapeTool) st = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
        Handle(XSControl_TransferReader) tr = reader.ChangeReader().WS()->TransferReader();
        TDF_LabelSequence free;
        st->GetFreeShapes(free);
        auto* out = new PgStep();
        for (int i = 1; i <= free.Length(); ++i) walk(st, tr, free.Value(i), TopLoc_Location(), deflection, out, 0);
        // make names unique
        std::map<std::string, int> seen;
        for (auto& b : out->bodies) {
            int n = ++seen[b.name];
            if (n > 1) b.name += "_" + std::to_string(n);
        }
        app->Close(doc);
        return out;
    } catch (const Standard_Failure& e) {
        set_err(err, err_len, std::string("OpenCASCADE: ") + e.GetMessageString());
        return nullptr;
    } catch (...) {
        set_err(err, err_len, "unknown error reading STEP");
        return nullptr;
    }
}

int pg_step_body_count(PgStep* s) { return (int)s->bodies.size(); }
const char* pg_step_body_name(PgStep* s, int i) { return s->bodies[i].name.c_str(); }
int pg_step_body_nverts(PgStep* s, int i) { return (int)(s->bodies[i].verts.size() / 3); }
const double* pg_step_body_verts(PgStep* s, int i) { return s->bodies[i].verts.data(); }
int pg_step_body_ntris(PgStep* s, int i) { return (int)(s->bodies[i].tris.size() / 3); }
const uint32_t* pg_step_body_tris(PgStep* s, int i) { return s->bodies[i].tris.data(); }
void pg_step_free(PgStep* s) { delete s; }

// Writes an assembly "toolhead" with components "duct" (hollow 10x10x40 box, 6x6 channel along X)
// and "nozzle" (10 mm cube). The duct instance is translated by +5 mm in Z.
int pg_step_write_demo(const char* path, char* err, int err_len) {
    try {
        Handle(XCAFApp_Application) app = XCAFApp_Application::GetApplication();
        Handle(TDocStd_Document) doc;
        app->NewDocument("MDTV-XCAF", doc);
        Handle(XCAFDoc_ShapeTool) st = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
        TopoDS_Shape outer = BRepPrimAPI_MakeBox(gp_Pnt(0, -5, -5), gp_Pnt(40, 5, 5)).Shape();
        TopoDS_Shape inner = BRepPrimAPI_MakeBox(gp_Pnt(-1, -3, -3), gp_Pnt(41, 3, 3)).Shape();
        TopoDS_Shape duct = BRepAlgoAPI_Cut(outer, inner).Shape();
        TopoDS_Shape nozzle = BRepPrimAPI_MakeBox(gp_Pnt(46, -5, -5), gp_Pnt(56, 5, 5)).Shape();
        TDF_Label ld = st->AddShape(duct, Standard_False);
        TDF_Label ln = st->AddShape(nozzle, Standard_False);
        TDataStd_Name::Set(ld, "duct");
        TDataStd_Name::Set(ln, "nozzle");
        {
            TopoDS_Shape left = BRepPrimAPI_MakeBox(gp_Pnt(100, 0, 0), gp_Pnt(105, 5, 5)).Shape();
            TopoDS_Shape right = BRepPrimAPI_MakeBox(gp_Pnt(110, 0, 0), gp_Pnt(115, 5, 5)).Shape();
            BRep_Builder bb;
            TopoDS_Compound comp;
            bb.MakeCompound(comp);
            bb.Add(comp, left);
            bb.Add(comp, right);
            TDF_Label lt = st->AddShape(comp, Standard_False);
            TDataStd_Name::Set(lt, "twin");
            TDF_Label sl, sr;
            st->AddSubShape(lt, left, sl);
            st->AddSubShape(lt, right, sr);
            if (!sl.IsNull()) TDataStd_Name::Set(sl, "Body_left");
            if (!sr.IsNull()) TDataStd_Name::Set(sr, "Body_right");
        }
        TDF_Label as = st->NewShape();
        TDataStd_Name::Set(as, "toolhead");
        gp_Trsf t;
        t.SetTranslation(gp_Vec(0, 0, 5));
        st->AddComponent(as, ld, TopLoc_Location(t));
        st->AddComponent(as, ln, TopLoc_Location());
        st->UpdateAssemblies();
        STEPCAFControl_Writer w;
        w.SetNameMode(Standard_True);
        if (!w.Transfer(doc, STEPControl_AsIs)) {
            set_err(err, err_len, "transfer failed");
            return 1;
        }
        if (w.Write(path) != IFSelect_RetDone) {
            set_err(err, err_len, "write failed");
            return 1;
        }
        app->Close(doc);
        return 0;
    } catch (const Standard_Failure& e) {
        set_err(err, err_len, std::string("OpenCASCADE: ") + e.GetMessageString());
        return 1;
    }
}

}
