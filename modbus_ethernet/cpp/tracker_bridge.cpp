#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
struct Run { int64_t row; int32_t a, b; };
struct Track {
    int64_t id, lo, last;
    int32_t x0, x1;
    int64_t area = 0;
    bool invalid = false;
    std::vector<Run> runs;
    std::vector<Run> frontier, next;
};
struct Group { int32_t a,b; std::vector<Run> raw; };
struct Engine {
    int gap, link, min_run, max_rows;
    bool index;
    Engine(int g,int l,int m,int max,bool idx):gap(g),link(l),min_run(m),max_rows(max),index(idx){}
    int64_t next_id=0, merged=0, indexed_rows=0, linear_rows=0;
    int64_t index_candidates=0, linear_candidates=0;
    std::vector<std::shared_ptr<Track>> active;
    std::vector<std::shared_ptr<Track>> finished;
    size_t run_count=0;
};
void error(char* out,size_t n,const char* message) {
    if(out && n) std::snprintf(out,n,"%s",message);
}
bool near(int a,int b,const std::vector<Run>& frontier,int link) {
    for(const auto& r:frontier) if(a<=r.b+link && r.a<=b+link) return true;
    return false;
}
void step(Engine& e,int64_t row,const uint8_t* hits,int width,int x0) {
    std::vector<Group> groups;
    for(int col=0;col<width;) {
        while(col<width && !hits[col]) ++col;
        int a=col;
        while(col<width && hits[col]) ++col;
        int b=col;
        if(b-a<e.min_run) continue;
        int aa=a+x0,bb=b+x0;
        if(!groups.empty() && aa<=groups.back().b+e.link) {
            groups.back().b=bb;
            groups.back().raw.push_back({row,aa,bb});
        } else groups.push_back({aa,bb,{{row,aa,bb}}});
    }
    for(auto& t:e.active) t->next.clear();
    bool indexed=e.index && e.active.size()+groups.size()>=24;
    if(indexed) ++e.indexed_rows; else ++e.linear_rows;
    for(const auto& g:groups) {
        std::vector<std::shared_ptr<Track>> matches;
        for(const auto& t:e.active) {
            // Lifetime bounds contain every frontier; this only skips impossible matches.
            if(indexed && (g.a>t->x1+e.link || t->x0>g.b+e.link)) continue;
            if(indexed) ++e.index_candidates; else ++e.linear_candidates;
            if(near(g.a,g.b,t->frontier,e.link) ||
               near(g.a,g.b,t->next,e.link)) matches.push_back(t);
        }
        std::shared_ptr<Track> t;
        if(matches.empty()) {
            t=std::make_shared<Track>();
            t->id=++e.next_id;t->lo=t->last=row;t->x0=g.a;t->x1=g.b;
            e.active.push_back(t);
        } else {
            t=matches.front();
            for(size_t i=1;i<matches.size();++i) {
                auto& other=matches[i];
                t->lo=std::min(t->lo,other->lo);
                t->last=std::max(t->last,other->last);
                t->x0=std::min(t->x0,other->x0);
                t->x1=std::max(t->x1,other->x1);
                t->area+=other->area;
                t->frontier.insert(t->frontier.end(),other->frontier.begin(),other->frontier.end());
                t->next.insert(t->next.end(),other->next.begin(),other->next.end());
                t->invalid=t->invalid || other->invalid;
                if(t->invalid) t->runs.clear();
                else t->runs.insert(t->runs.end(),other->runs.begin(),other->runs.end());
                e.active.erase(std::find(e.active.begin(),e.active.end(),other));
                ++e.merged;
            }
        }
        t->x0=std::min(t->x0,g.a);t->x1=std::max(t->x1,g.b);
        t->next.push_back({row,g.a,g.b});
        for(const auto& run:g.raw) t->area+=run.b-run.a;
        if(row-t->lo+1>e.max_rows) {t->invalid=true;t->runs.clear();}
        if(!t->invalid) t->runs.insert(t->runs.end(),g.raw.begin(),g.raw.end());
    }
    std::vector<std::shared_ptr<Track>> keep;
    for(auto& t:e.active) {
        if(!t->next.empty()) {t->frontier=t->next;t->last=row;}
        if(row-t->last>=e.gap) {
            e.run_count+=t->runs.size();
            if(e.run_count>500000) throw std::runtime_error("too many finished runs");
            e.finished.push_back(t);
        } else keep.push_back(t);
    }
    e.active=std::move(keep);
}
} // namespace

extern "C" {
struct JellyTrack {
    int64_t id, lo, last;
    int32_t x0, x1;
    int64_t area;
    int32_t invalid;
    size_t run_offset, run_count;
};
struct JellyTrackerStats {
    size_t count, run_count, active;
    int64_t merged, indexed_rows, linear_rows, index_candidates, linear_candidates;
};
int jelly_tracker_create(int gap,int link,int min_run,int max_rows,int index,void** handle,
                         char* err,size_t cap) {
    try {
        if(!handle || gap<1 || link<0 || min_run<1 || max_rows<1)
            throw std::runtime_error("invalid tracker config");
        *handle=new Engine(gap,link,min_run,max_rows,index!=0);
        return 0;
    } catch(const std::exception& ex) {error(err,cap,ex.what());return -1;}
}
int jelly_tracker_feed(void* handle,const uint8_t* hits,int rows,int width,int64_t row0,
                       int x0,JellyTrackerStats* stats,char* err,size_t cap) {
    try {
        auto* e=static_cast<Engine*>(handle);
        if(!e || !hits || !stats || rows<1 || rows>256 || width<1 || width>4096 ||
           !e->finished.empty()) throw std::runtime_error("invalid tracker feed or pending output");
        for(int i=0;i<rows;++i) step(*e,row0+i,hits+static_cast<size_t>(i)*width,width,x0);
        *stats={e->finished.size(),e->run_count,e->active.size(),e->merged,
                e->indexed_rows,e->linear_rows,e->index_candidates,e->linear_candidates};
        return 0;
    } catch(const std::exception& ex) {error(err,cap,ex.what());return -1;}
}
int jelly_tracker_export(void* handle,JellyTrack* tracks,size_t track_capacity,
                         int64_t* runs,size_t run_capacity,char* err,size_t cap) {
    try {
        auto* e=static_cast<Engine*>(handle);
        if(!e || !tracks || !runs || track_capacity<e->finished.size() ||
           run_capacity<e->run_count*3) throw std::runtime_error("tracker export buffer too small");
        size_t pos=0;
        for(size_t i=0;i<e->finished.size();++i) {
            const auto& t=*e->finished[i];
            tracks[i]={t.id,t.lo,t.last,t.x0,t.x1,t.area,static_cast<int>(t.invalid),pos,t.runs.size()};
            for(const auto& r:t.runs) {runs[3*pos]=r.row;runs[3*pos+1]=r.a;runs[3*pos+2]=r.b;++pos;}
        }
        e->finished.clear();e->run_count=0;
        return 0;
    } catch(const std::exception& ex) {error(err,cap,ex.what());return -1;}
}
size_t jelly_tracker_reset(void* handle) {
    auto* e=static_cast<Engine*>(handle);
    if(!e) return 0;
    size_t n=e->active.size();e->active.clear();e->finished.clear();e->run_count=0;
    return n;
}
void jelly_tracker_destroy(void* handle) {delete static_cast<Engine*>(handle);}
}
