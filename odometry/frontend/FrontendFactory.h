#pragma once
#include <memory>
#include "IFrontend.h"
#include "../OdometryTypes.h"

// Builds the frontend named by OdometryConfig::frontend_type. Defaults to the
// descriptor frontend (detect+match). The optical_flow frontend is added in
// OF-1; until then an unknown type falls back to descriptor.
class FrontendFactory {
public:
    static std::unique_ptr<IFrontend> create(const OdometryConfig& config);
};
