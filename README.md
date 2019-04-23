# File Connector

The File Connector enables users to access files as objects using the Swift API.
It has been packaged as a docker container to provide easy deployment.

Files and directories are exposed following the same path hierarchy as they are
laid out on the Filesystem. For example, an NFS share mounted under
``/mnt/nfs_share/dir/foo`` could be accessed with the URL
``http://host:port/fileconnector/nfs_share/dir/foo``.

## Use Cases

The File Connector can be especially useful together with
[1space](http://github.com/swiftstack/1space) to move data between a NAS share
and a Swift cluster. 

## Metadata persistence

The File Connector has two modes of metadata persistence. When targetting
Filesystems that support extended attributes (i.e. xattr), metadata can be
saved together with with the file as extended attributes (This is similar to
how Openstack Swift persists metadata).

Another mode developed for Filesystems without xattr support is to save
metadata in hidden ``json`` files. For example:
The file ``/mnt/nfs_share/dir/foo`` will have its metadata saved under ``/mnt/nfs_share/dir/.fc_meta/foo/obj_metadata.json``

## How-To run

File Connector is packaged as a docker container. You can run it as a single
instance or in clustered mode. There are sample Compose files
under the ``examples`` directory to quickly get started or read below for
details on the different options that can be set to run.

### Single instance mode

Just run the docker container, substituting the configuration between brackets
``<>``:

```
docker run -d \
  -v </path/to/sharename>:/srv/fileconnector/<sharename> \
  -p 8083:8083 \
  -e FILE_CONNECTOR_SECRET=<choose_your_secret> \
  --name file-connector \
  swiftstack/file-connector
```

You can now access a NAS share using the [swift client](https://github.com/openstack/python-swiftclient)
for example. The credentials are: account: ``fileconnector``,
 user: ``fcuser``, password: ``<choose_your_secret>``

### Clustered mode

To run in clustered mode you will need to run a load balancer and an instance
of memcached. Then just launch as many File Connector containers as you would
like, making sure to update the port mapping but re-use the same
``FILE_CONNECTOR_SECRET`` for all containers. You will also need to pass a new
environment variable to set the memcache server in the File Connector.

Refer to [examples/cluster](examples/cluster) as an example of how
to run all services as docker containers.
