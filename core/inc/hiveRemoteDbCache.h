#ifndef HIVEREMOTEDBCACHE_H
#define HIVEREMOTEDBCACHE_H

void hiveRemoteInitDbLookupTables();
inline void hiveRemoteAddGuidLocalCopy( hiveGuid_t guid, void * ptr  );
inline void * hiveRemoteGetGuidLocalCopy( hiveGuid_t guid  );
inline void hiveRemoteAddGuidToSentList( hiveGuid_t guid, int rank  );
inline bool hiveRemoteGuidSentCopyExists( hiveGuid_t guid, int rank  );

#endif
