#include "ConfigLoader.h"

ConfigLoader::ConfigLoader(const std::string& yaml_path)
    : fs(yaml_path, cv::FileStorage::READ) {}

bool ConfigLoader::loadOdometryConfig(OdometryConfig& out) {
    if (!fs.isOpened()) return false;

    // Detector / matcher / pose-estimator selection
    if (!fs["detector"]["type"].empty()) {
        out.detector_type = (std::string)fs["detector"]["type"];
    }
    if (!fs["matcher"]["type"].empty()) {
        out.matcher_type = (std::string)fs["matcher"]["type"];
    }
    if (!fs["pose_estimator"]["type"].empty()) {
        out.pose_estimator_type = (std::string)fs["pose_estimator"]["type"];
    }

    // Keyframing caps.
    cv::FileNode kf = fs["keyframe"];
    if (!kf.empty()) {
        if (!kf["max_skip"].empty())    out.keyframe_max_skip    = (int)kf["max_skip"];
        if (!kf["min_matches"].empty()) out.keyframe_min_matches = (int)kf["min_matches"];
    }

    // Detector params (per-type block under detector.<type>)
    cv::FileNode d_node = fs["detector"][out.detector_type];
    if (!d_node.empty()) {
        if (out.detector_type == "ORB") {
            out.detector_params["nfeatures"]    = (float)(int)d_node["nfeatures"];
            out.detector_params["scale_factor"] = (float)d_node["scaleFactor"];
            out.detector_params["nlevels"]      = (float)(int)d_node["nLevels"];
            // Optional ORB knobs. Read only if present, so callers that
            // omit them keep the cv::ORB::create defaults.
            if (!d_node["edgeThreshold"].empty()) out.detector_params["edge_threshold"] = (float)(int)d_node["edgeThreshold"];
            if (!d_node["firstLevel"].empty())    out.detector_params["first_level"]    = (float)(int)d_node["firstLevel"];
            if (!d_node["WTA_K"].empty())         out.detector_params["wta_k"]          = (float)(int)d_node["WTA_K"];
            if (!d_node["patchSize"].empty())     out.detector_params["patch_size"]     = (float)(int)d_node["patchSize"];
            if (!d_node["fastThreshold"].empty()) out.detector_params["fast_threshold"] = (float)(int)d_node["fastThreshold"];
        } else if (out.detector_type == "SIFT") {
            out.detector_params["nfeatures"]         = (float)(int)d_node["nfeatures"];
            out.detector_params["nOctaveLayers"]     = (float)(int)d_node["nOctaveLayers"];
            out.detector_params["contrastThreshold"] = (float)d_node["contrastThreshold"];
            out.detector_params["edgeThreshold"]     = (float)(int)d_node["edgeThreshold"];
            out.detector_params["sigma"]             = (float)d_node["sigma"];
        }
    }

    // Matcher params
    cv::FileNode m_node = fs["matcher"];
    if (!m_node.empty()) {
        if (!m_node["distance_ratio"].empty()) {
            out.matcher_params["ratio_thresh"] = (float)m_node["distance_ratio"];
        }
        if (out.matcher_type == "FLANN" && !m_node["FLANN"].empty()) {
            out.matcher_params["kdTrees"]      = (float)(int)m_node["FLANN"]["kdTrees"];
            out.matcher_params["searchChecks"] = (float)(int)m_node["FLANN"]["searchChecks"];
        }
    }

    // Bucketing
    cv::FileNode b_node = fs["bucketing"];
    if (!b_node.empty()) {
        out.bucketing_params.enabled                 = (int)b_node["enabled"] != 0;
        out.bucketing_params.grid_cols               = (int)b_node["grid_cols"];
        out.bucketing_params.grid_rows               = (int)b_node["grid_rows"];
        out.bucketing_params.max_features_per_bucket = (int)b_node["max_features_per_bucket"];
    }

    // Local bundle adjustment. Only `enabled` is gated; the rest of the
    // sub-keys are overlaid onto the LBAParams defaults if present.
    cv::FileNode lba_node = fs["local_bundle_adjustment"];
    if (!lba_node.empty()) {
        out.use_local_ba = (int)lba_node["enabled"] != 0;

        LBAParams& lp = out.lba_params;
        auto readD = [&](const char* k, double& dst) {
            if (!lba_node[k].empty()) dst = (double)lba_node[k];
        };
        auto readI = [&](const char* k, int& dst) {
            if (!lba_node[k].empty()) dst = (int)lba_node[k];
        };

        readI("window_size",                lp.window_size);
        readI("opt_stride",                 lp.opt_stride);
        readD("anchor_prior_sigma",         lp.anchor_prior_sigma);
        readD("end_prior_rot_sigma",        lp.end_prior_rot_sigma);
        readD("end_prior_trans_sigma",      lp.end_prior_trans_sigma);
        readD("between_rot_sigma",          lp.between_rot_sigma);
        readD("between_trans_sigma",        lp.between_trans_sigma);
        readD("stationary_rot_sigma",       lp.stationary_rot_sigma);
        readD("stationary_trans_sigma",     lp.stationary_trans_sigma);
        readD("pixel_sigma",                lp.pixel_sigma);
        readD("rank_tolerance",             lp.rank_tolerance);
        readD("outlier_threshold",          lp.outlier_threshold);
        readI("min_observations",           lp.min_observations);
        readD("min_bbox_diagonal",          lp.min_bbox_diagonal);
        readI("min_smart_factors",          lp.min_smart_factors);
        readD("max_correction_translation", lp.max_correction_translation);
        readD("max_correction_rotation_deg",lp.max_correction_rotation_deg);
    }

    // System (threads + backend + verbose)
    cv::FileNode sys_node = fs["system"];
    if (!sys_node.empty()) {
        if (!sys_node["num_threads"].empty()) {
            out.num_threads = (int)sys_node["num_threads"];
        }
        if (!sys_node["backend"].empty()) {
            std::string s = (std::string)sys_node["backend"];
            if (s == "CUDA")        out.backend = ComputeBackend::CUDA;
            else if (s == "OPENCL") out.backend = ComputeBackend::OPENCL;
            else                    out.backend = ComputeBackend::CPU;
        }
        if (!sys_node["verbose"].empty()) {
            out.verbose = (int)sys_node["verbose"] != 0;
        }
    }

    // Camera intrinsics. Primary key is top-level "camera:"; fall back to the
    // legacy "airsim.intrinsics:" block so older YAML configs keep working.
    cv::FileNode cam_node = fs["camera"];
    if (cam_node.empty()) {
        cam_node = fs["airsim"]["intrinsics"];
    }
    if (!cam_node.empty()) {
        out.intrinsics.fx = (float)cam_node["fx"];
        out.intrinsics.fy = (float)cam_node["fy"];
        out.intrinsics.cx = (float)cam_node["cx"];
        out.intrinsics.cy = (float)cam_node["cy"];
    }

    // pose_params.min_disparity used to be set inline by every main. Seed it
    // here if the caller didn't supply one.
    if (!out.pose_params.count("min_disparity")) {
        out.pose_params["min_disparity"] = 2.0f;
    }

    return true;
}

bool ConfigLoader::loadKittiDataset(std::string& out_root, std::string& out_sequence) {
    if (!fs.isOpened()) return false;
    cv::FileNode n = fs["dataset"];
    if (n.empty()) return false;
    out_root     = (std::string)n["root_path"];
    out_sequence = (std::string)n["sequence"];
    return true;
}

bool ConfigLoader::loadPlaybackVelocity(float& out_velocity) {
    if (!fs.isOpened()) return false;
    cv::FileNode airsim = fs["airsim"];
    if (airsim.empty() || airsim["playback_velocity"].empty()) return false;
    out_velocity = (float)airsim["playback_velocity"];
    return true;
}

bool ConfigLoader::loadRosbagConfig(RosbagConfig& out) {
    if (!fs.isOpened()) return false;
    cv::FileNode n = fs["rosbag"];
    if (n.empty() || n["bag_path"].empty()) return false;

    out.bag_path = (std::string)n["bag_path"];
    if (!n["img_topic"].empty())  out.img_topic = (std::string)n["img_topic"];
    if (!n["gps_topic"].empty())  out.gps_topic = (std::string)n["gps_topic"];
    if (!n["imu_topic"].empty())  out.imu_topic = (std::string)n["imu_topic"];
    if (!n["ppk_path"].empty())   out.ppk_path  = (std::string)n["ppk_path"];
    if (!n["start_time"].empty()) out.start_time = (double)n["start_time"];
    if (!n["end_time"].empty())   out.end_time   = (double)n["end_time"];

    // cam0->body extrinsic (top-level opencv-matrix). Left empty when absent,
    // which the evaluator treats as identity.
    cv::FileNode ext = fs["body_T_cam0"];
    if (!ext.empty()) ext >> out.body_T_cam0;

    // Per-axis sign flip for the logged trajectory, to reconcile a VO-vs-ENU
    // handedness mismatch (a reflection a rotation cannot fix). Defaults to no
    // flip; each axis is read independently.
    cv::FileNode flip = fs["trajectory_flip"];
    if (!flip.empty()) {
        if (!flip["x"].empty()) out.traj_sign[0] = (double)flip["x"];
        if (!flip["y"].empty()) out.traj_sign[1] = (double)flip["y"];
        if (!flip["z"].empty()) out.traj_sign[2] = (double)flip["z"];
    }

    cv::FileNode viz = fs["visualizer"];
    if (!viz.empty() && !viz["scale"].empty()) out.viz_scale = (double)viz["scale"];

    cv::FileNode al = fs["aligner"];
    if (!al.empty() && !al["init_distance"].empty())
        out.aligner_init_distance = (double)al["init_distance"];
    return true;
}

bool ConfigLoader::loadCameraModel(CameraModelConfig& out) {
    if (!fs.isOpened()) return false;
    cv::FileNode cam = fs["camera"];
    if (cam.empty() || cam["model_type"].empty()) return false;

    out.model_type = (std::string)cam["model_type"];
    if (!cam["scale_factor"].empty()) out.scale_factor = (double)cam["scale_factor"];
    if (!cam["image_width"].empty())  out.image_width  = (int)cam["image_width"];
    if (!cam["image_height"].empty()) out.image_height = (int)cam["image_height"];
    out.mu = (double)cam["mu"];
    out.mv = (double)cam["mv"];
    out.u0 = (double)cam["u0"];
    out.v0 = (double)cam["v0"];
    if (!cam["k2"].empty()) out.k2 = (double)cam["k2"];
    if (!cam["k3"].empty()) out.k3 = (double)cam["k3"];
    if (!cam["k4"].empty()) out.k4 = (double)cam["k4"];
    if (!cam["k5"].empty()) out.k5 = (double)cam["k5"];
    return true;
}
